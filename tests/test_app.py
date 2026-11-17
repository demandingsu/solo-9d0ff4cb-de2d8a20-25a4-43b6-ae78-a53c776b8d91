"""关键情形自测:拆分正确性、纠错规则、损坏处理、原子发布、恢复、不可改写。"""
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.demux import (  # noqa: E402
    FastqFormatError,
    assign_index,
    build_matcher,
    iter_fastq,
    read_id,
)
from app.samples import SampleSheetError, load_samples  # noqa: E402

SAMPLES_CSV = "sample_id,index\nS1,AAAACCCC\nS2,AAAACCGG\nS3,GGGGTTTT\n"

# S1=AAAACCCC 与 S2=AAAACCGG 相距 2,便于构造歧义错配;S3 远离二者
READS = [
    ("r001", "AAAACCCC"),  # S1 精确
    ("r002", "AAAACCGG"),  # S2 精确
    ("r003", "AAAACCCA"),  # 距 S1 唯一单碱基错配 -> 纠错到 S1
    ("r004", "AAAACCGC"),  # 距 S1、S2 各 1 -> 歧义 -> 未分配
    ("r005", "NNNNNNNN"),  # 无候选 -> 未分配
    ("r006", "GGGGTTTA"),  # 距 S3 唯一单碱基错配 -> 纠错到 S3
    ("r007", "AAAACCCC"),  # S1 精确(检验同组内顺序)
]

SEQ1, SEQ2 = "ACGT" * 10, "TGCA" * 10
QUAL = "I" * 40


def fq(records):
    return "".join(f"@{h}\n{s}\n+\n{q}\n" for h, s, q in records)


def make_inputs(reads=READS):
    r1, r2, i1 = [], [], []
    for n, (rid, idx) in enumerate(reads):
        head = f"INST:1:FC:1:1101:{1000 + n}:{2000 + n}"
        r1.append((f"{head} 1:N:0:0", SEQ1, QUAL))
        r2.append((f"{head} 2:N:0:0", SEQ2, QUAL))
        i1.append((f"{head} 1:N:0:0", idx, "I" * len(idx)))
    return fq(r1).encode(), fq(r2).encode(), fq(i1).encode(), SAMPLES_CSV.encode()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        c.data_dir = tmp_path / "data"
        yield c


def submit(client, run_number, r1, r2, i1, samples):
    return client.post(
        "/runs",
        data={"run_number": run_number},
        files={
            "r1": ("r1.fastq", r1),
            "r2": ("r2.fastq", r2),
            "i1": ("i1.fastq", i1),
            "samples": ("samples.csv", samples),
        },
        follow_redirects=False,
    )


def submit_good(client, run_number="RUN1"):
    r1, r2, i1, csv = make_inputs()
    return submit(client, run_number, r1, r2, i1, csv)


def run_id_of(resp):
    assert resp.status_code == 303, resp.text
    return int(resp.headers["location"].rsplit("/", 1)[1])


def wait_done(client, run_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/api/runs/{run_id}").json()
        if st["status"] != "processing":
            return st
        time.sleep(0.05)
    raise TimeoutError("处理超时")


def parse_fastq_ids(text):
    lines = text.splitlines()
    return [lines[i][1:].split()[0] for i in range(0, len(lines), 4)]


def head_of(n):
    """第 n 条 READS 记录对应的读段号(见 make_inputs)。"""
    return f"INST:1:FC:1:1101:{1000 + n}:{2000 + n}"


def counts_of(status):
    return {c["sample_id"]: (c["pairs"], c["corrected"]) for c in status["counts"]}


# ------------------------------------------------------------ 单元级

def test_assign_index_rules():
    samples = [("S1", "AAAACCCC"), ("S2", "AAAACCGG"), ("S3", "GGGGTTTT")]
    exact, neighbors = build_matcher(samples)
    assert assign_index("AAAACCCC", exact, neighbors) == ("S1", False)   # 精确优先
    assert assign_index("AAAACCCA", exact, neighbors) == ("S1", True)    # 唯一错配
    assert assign_index("AAAACCGC", exact, neighbors) == (None, False)   # 歧义
    assert assign_index("NNNNNNNN", exact, neighbors) == (None, False)   # 无候选
    assert assign_index("AAAA", exact, neighbors) == (None, False)       # 长度不符


def test_read_id():
    assert read_id("INST:1:FC:1:1101:100:100 1:N:0:0") == "INST:1:FC:1:1101:100:100"
    assert read_id("PLAINID") == "PLAINID"
    assert read_id("") == ""


def test_iter_fastq_truncated(tmp_path):
    p = tmp_path / "bad.fastq"
    p.write_text("@a\nACGT\n+\n")  # 缺质量行
    with pytest.raises(FastqFormatError, match="截断"):
        list(iter_fastq(str(p), "R1"))


def test_iter_fastq_len_mismatch(tmp_path):
    p = tmp_path / "bad.fastq"
    p.write_text("@a\nACGT\n+\nIII\n")  # 质量短 1
    with pytest.raises(FastqFormatError, match="长度"):
        list(iter_fastq(str(p), "R1"))


def test_samples_validation(tmp_path):
    def load(text):
        p = tmp_path / "s.csv"
        p.write_text(text)
        return load_samples(str(p))

    assert load("sid,idx\nA,ACGT\n") == [("A", "ACGT")]
    assert load("A,acgt\n") == [("A", "ACGT")]  # 小写归一、无表头
    with pytest.raises(SampleSheetError, match="重复"):
        load("A,ACGT\nB,ACGT\n")
    with pytest.raises(SampleSheetError, match="长度不一致"):
        load("A,ACGT\nB,ACGTA\n")
    with pytest.raises(SampleSheetError, match="非法字符"):
        load("A,ACGT\nB,ACGB\n")
    with pytest.raises(SampleSheetError, match="保留字"):
        load("undetermined,ACGT\n")


# ------------------------------------------------------------ 端到端

def test_happy_path(client):
    rid = run_id_of(submit_good(client))
    st = wait_done(client, rid)
    assert st["status"] == "published", st["error"]
    assert st["total_pairs"] == 7
    assert counts_of(st) == {
        "S1": (3, 1), "S2": (1, 0), "S3": (1, 1), "undetermined": (2, 0),
    }

    s1_r1 = client.get(f"/runs/{rid}/download/S1_R1.fastq")
    assert s1_r1.status_code == 200
    assert parse_fastq_ids(s1_r1.text) == [head_of(0), head_of(2), head_of(6)]  # 原顺序
    s1_r2 = client.get(f"/runs/{rid}/download/S1_R2.fastq")
    assert parse_fastq_ids(s1_r2.text) == [head_of(0), head_of(2), head_of(6)]
    assert SEQ2 in s1_r2.text  # R2 内容确实是 R2
    und = client.get(f"/runs/{rid}/download/undetermined_R1.fastq")
    assert parse_fastq_ids(und.text) == [head_of(3), head_of(4)]

    log = client.get(f"/runs/{rid}/download/demux.log")
    assert log.status_code == 200 and "总读段对: 7" in log.text

    page = client.get(f"/runs/{rid}")
    assert page.status_code == 200 and "AAAACCCC" in page.text and "已发布" in page.text


def test_truncated_input_fails_then_retry_succeeds(client):
    r1, r2, i1, csv = make_inputs()
    bad_r2 = r2[: r2.rfind(b"\n", 0, r2.rfind(b"\n"))]  # 末条记录缺质量行
    rid = run_id_of(submit(client, "RUN2", r1, bad_r2, i1, csv))
    st = wait_done(client, rid)
    assert st["status"] == "failed"
    assert "截断" in st["error"]
    # 不发布半套文件
    assert not (client.data_dir / "published" / "RUN2").exists()
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 409

    # 同一运行号重试可完成
    rid2 = run_id_of(submit_good(client, "RUN2"))
    st2 = wait_done(client, rid2)
    assert st2["status"] == "published"
    assert st2["total_pairs"] == 7
    assert client.get(f"/runs/{rid2}/download/S1_R1.fastq").status_code == 200


def test_read_id_mismatch_fails(client):
    r1, r2, i1, csv = make_inputs()
    bad_r2 = r2.replace(b"1101:1001:2001", b"1101:9999:2001", 1)
    rid = run_id_of(submit(client, "RUN3", r1, bad_r2, i1, csv))
    st = wait_done(client, rid)
    assert st["status"] == "failed" and "读段号不一致" in st["error"]
    assert not (client.data_dir / "published" / "RUN3").exists()


def test_seq_qual_len_mismatch_fails(client):
    r1, r2, i1, csv = make_inputs()
    bad_i1 = i1.replace(b"IIIIIIII\n", b"IIIIIII\n", 1)  # 首条 I1 质量少 1
    rid = run_id_of(submit(client, "RUN4", r1, r2, bad_i1, csv))
    st = wait_done(client, rid)
    assert st["status"] == "failed" and "长度" in st["error"]


def test_record_count_mismatch_fails(client):
    r1, r2, i1, csv = make_inputs()
    extra = f"@INST:1:FC:1:1101:9:9 1:N:0:0\n{SEQ1}\n+\n{QUAL}\n".encode()
    rid = run_id_of(submit(client, "RUN5", r1 + extra, r2, i1, csv))
    st = wait_done(client, rid)
    assert st["status"] == "failed" and "记录数不一致" in st["error"]


def test_published_run_immutable(client):
    rid = run_id_of(submit_good(client, "RUN6"))
    assert wait_done(client, rid)["status"] == "published"
    before = client.get(f"/runs/{rid}/download/S1_R1.fastq").text

    resp = submit_good(client, "RUN6")  # 同号再提交
    assert resp.status_code == 409
    assert "不可改写" in resp.json()["detail"]
    after = client.get(f"/runs/{rid}/download/S1_R1.fastq").text
    assert before == after  # 文件未被改写


def test_bad_sample_sheet_rejected(client):
    r1, r2, i1, _ = make_inputs()
    dup = b"A,ACGT\nB,ACGT\n"
    resp = submit(client, "RUN7", r1, r2, i1, dup)
    assert resp.status_code == 400
    assert "重复" in resp.json()["detail"]


def test_interrupted_run_recovered_on_startup(client):
    """模拟进程中断:遗留 processing 记录 + 半成品目录,重启后标失败、清理干净。"""
    from app.main import recover_interrupted
    from app.db import db_conn
    from app.config import db_path, staging_dir, published_dir

    staging = staging_dir() / "deadbeef"
    (staging / "out").mkdir(parents=True)
    (staging / "out" / "S1_R1.fastq").write_text("半成品")
    stray_pub = published_dir() / "RUN8"
    stray_pub.mkdir(parents=True)
    (stray_pub / "S1_R1.fastq").write_text("改名后来不及落库的文件")
    with db_conn(db_path()) as conn:
        conn.execute(
            "INSERT INTO runs(run_number, status, staging_dir, created_at, updated_at) "
            "VALUES ('RUN8', 'processing', ?, 't', 't')",
            (str(staging),),
        )

    recover_interrupted()  # 模拟重启

    with db_conn(db_path()) as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_number='RUN8'").fetchone()
    assert row["status"] == "failed" and "中断" in row["error"]
    assert not staging.exists()
    assert not stray_pub.exists()
    # 恢复后同一运行号可重试并完成
    rid = run_id_of(submit_good(client, "RUN8"))
    assert wait_done(client, rid)["status"] == "published"


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "新建拆样运行" in resp.text


# ------------------------------------------------------------ 质控(Q20 占比暂扣/确认放行)

def make_qc_inputs():
    """S1 的 R1 全部低质量(>10% 低于 Q20)将被暂扣;S2 高质量放行;
    S3 无读段(空组);一条未分配读段。"""
    samples_csv = "sample_id,index\nS1,AAAACCCC\nS2,AAAACCGG\nS3,GGGGTTTT\n"
    reads = (
        [("AAAACCCC", "#" * 40, "I" * 40)] * 10   # S1:R1 全为 Q2
        + [("AAAACCGG", "I" * 40, "I" * 40)] * 10  # S2:全部 Q40
        + [("NNNNNNNN", "I" * 40, "I" * 40)]       # 未分配
    )
    r1, r2, i1 = [], [], []
    for n, (idx, q1, q2) in enumerate(reads):
        head = f"INST:1:FC:1:1101:{3000 + n}:{4000 + n}"
        r1.append((f"{head} 1:N:0:0", SEQ1, q1))
        r2.append((f"{head} 2:N:0:0", SEQ2, q2))
        i1.append((f"{head} 1:N:0:0", idx, "I" * len(idx)))
    return fq(r1).encode(), fq(r2).encode(), fq(i1).encode(), samples_csv.encode()


def submit_qc(client, run_number="RUNQ"):
    r1, r2, i1, csv = make_qc_inputs()
    return submit(client, run_number, r1, r2, i1, csv)


def qc_of(status):
    return {c["sample_id"]: c["qc"] for c in status["counts"]}


def test_below_q20_phred33():
    from app.demux import below_q20
    assert below_q20("I" * 40) == 0    # Q40
    assert below_q20("5" * 40) == 0    # 恰为 Q20,不算"低于"
    assert below_q20("4" * 40) == 40   # Q19
    assert below_q20("#" * 40) == 40   # Q2
    assert below_q20("I5I4") == 1      # 仅 '4' 低于 Q20


def test_qc_verdict_boundaries():
    from app.main import qc_verdict
    assert qc_verdict(0, 0, 0, 0) == "no_data"        # 空组
    assert qc_verdict(100, 10, 100, 0) == "released"  # 恰好 10% 不算超过
    assert qc_verdict(100, 11, 100, 0) == "held"      # R1 超 10%
    assert qc_verdict(100, 0, 100, 11) == "held"      # R2 超 10%
    assert qc_verdict(100, 5, 100, 5) == "released"


def test_qc_hold_and_confirm_release(client):
    rid = run_id_of(submit_qc(client))
    st = wait_done(client, rid)
    assert st["status"] == "published", st["error"]
    qc = qc_of(st)
    assert qc["S1"]["status"] == "held" and not qc["S1"]["confirmed"]
    assert (qc["S1"]["r1_total"], qc["S1"]["r1_below"]) == (400, 400)
    assert (qc["S1"]["r2_total"], qc["S1"]["r2_below"]) == (400, 0)
    assert qc["S2"]["status"] == "released"
    assert qc["S3"]["status"] == "no_data"
    assert qc["undetermined"]["status"] == "released"

    # 暂扣组双端都不可下载;放行组可下载;空组无数据不可下载;日志不受限
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 409
    assert client.get(f"/runs/{rid}/download/S1_R2.fastq").status_code == 409
    assert client.get(f"/runs/{rid}/download/S2_R1.fastq").status_code == 200
    assert client.get(f"/runs/{rid}/download/S2_R2.fastq").status_code == 200
    assert client.get(f"/runs/{rid}/download/S3_R1.fastq").status_code == 409
    assert client.get(f"/runs/{rid}/download/undetermined_R1.fastq").status_code == 200
    assert client.get(f"/runs/{rid}/download/demux.log").status_code == 200

    # 运行页显示双端占比、状态与确认入口
    page = client.get(f"/runs/{rid}").text
    assert "100.00%" in page and "0.00%" in page
    assert "暂扣" in page and "确认放行" in page and "无数据" in page

    # 日志里的质量结论与同次文件、对数同批产生
    log = client.get(f"/runs/{rid}/download/demux.log").text
    assert "S1\tR1=100.00%\tR2=0.00%\t暂扣" in log
    assert "S3\t无数据" in log

    # 非暂扣组/不存在组不能确认
    assert client.post(f"/runs/{rid}/groups/S2/release").status_code == 409
    assert client.post(f"/runs/{rid}/groups/S3/release").status_code == 409
    assert client.post(f"/runs/{rid}/groups/NOPE/release").status_code == 404

    # 确认放行后双端可下载;重复确认幂等
    resp = client.post(f"/runs/{rid}/groups/S1/release", follow_redirects=False)
    assert resp.status_code == 303
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 200
    assert client.get(f"/runs/{rid}/download/S1_R2.fastq").status_code == 200
    again = client.post(f"/runs/{rid}/groups/S1/release", follow_redirects=False)
    assert again.status_code == 303  # 重复确认幂等
    st2 = client.get(f"/api/runs/{rid}").json()
    assert qc_of(st2)["S1"]["confirmed"] is True


def test_qc_confirm_survives_restart(client):
    from app.main import recover_interrupted
    rid = run_id_of(submit_qc(client, "RUNRS"))
    assert wait_done(client, rid)["status"] == "published"
    client.post(f"/runs/{rid}/groups/S1/release")
    recover_interrupted()  # 模拟重启:已发布组的确认仍有效
    st = client.get(f"/api/runs/{rid}").json()
    assert st["status"] == "published"
    assert qc_of(st)["S1"]["confirmed"] is True
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 200


def test_qc_concurrent_confirm_and_download(client):
    """并发确认与下载:每个请求只看到已提交的组状态,确认全部落库后必放行。"""
    import concurrent.futures
    rid = run_id_of(submit_qc(client, "RUNC"))
    assert wait_done(client, rid)["status"] == "published"

    def confirm():
        return client.post(f"/runs/{rid}/groups/S1/release",
                           follow_redirects=False).status_code

    def dl():
        return client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(confirm) if i % 2 == 0 else ex.submit(dl)
                for i in range(16)]
        results = [f.result() for f in futs]
    confirms = results[0::2]
    downloads = results[1::2]
    assert set(confirms) == {303}            # 并发确认幂等,都成功
    assert set(downloads) <= {200, 409}      # 只按已提交状态供数,无第三种结果
    # 确认全部提交后,下载必然放行
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 200
    st = client.get(f"/api/runs/{rid}").json()
    assert qc_of(st)["S1"]["confirmed"] is True


def test_qc_failed_run_retry_no_inheritance(client):
    """尾部损坏 -> 失败,不留质控结论;同号重试产生全新结论,不继承旧确认。"""
    r1, r2, i1, csv = make_qc_inputs()
    bad_r2 = r2[: r2.rfind(b"\n", 0, r2.rfind(b"\n"))]  # 末条记录缺质量行
    rid = run_id_of(submit(client, "RUNF", r1, bad_r2, i1, csv))
    st = wait_done(client, rid)
    assert st["status"] == "failed" and "截断" in st["error"]
    assert st["counts"] == []  # 失败运行没有任何计数与质控结论
    assert not (client.data_dir / "published" / "RUNF").exists()
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 409
    assert client.post(f"/runs/{rid}/groups/S1/release").status_code == 409

    rid2 = run_id_of(submit_qc(client, "RUNF"))  # 同号重试
    st2 = wait_done(client, rid2)
    assert st2["status"] == "published"
    qc = qc_of(st2)
    assert qc["S1"]["status"] == "held" and not qc["S1"]["confirmed"]
    assert client.get(f"/runs/{rid2}/download/S1_R1.fastq").status_code == 409
    client.post(f"/runs/{rid2}/groups/S1/release")
    assert client.get(f"/runs/{rid2}/download/S1_R1.fastq").status_code == 200


def test_legacy_run_without_qc_unevaluated(client):
    """旧运行没有质控记录:页面标未评估,下载保持原行为,不能确认。"""
    from app.db import db_conn
    from app.config import db_path, published_dir
    pub = published_dir() / "OLD1"
    pub.mkdir(parents=True)
    (pub / "S1_R1.fastq").write_text(fq([("h 1:N:0:0", SEQ1, QUAL)]))
    (pub / "S1_R2.fastq").write_text(fq([("h 2:N:0:0", SEQ2, QUAL)]))
    (pub / "demux.log").write_text("旧日志")
    with db_conn(db_path()) as conn:
        cur = conn.execute(
            "INSERT INTO runs(run_number, status, staging_dir, created_at, "
            "updated_at, published_at, total_pairs) "
            "VALUES ('OLD1', 'published', '', 't', 't', 't', 1)"
        )
        rid = cur.lastrowid
        conn.execute(
            "INSERT INTO sample_counts(run_id, sample_id, index_seq, pairs, "
            "corrected) VALUES (?, 'S1', 'AAAACCCC', 1, 0)",
            (rid,),
        )
    page = client.get(f"/runs/{rid}").text
    assert "未评估" in page
    assert client.get(f"/runs/{rid}/download/S1_R1.fastq").status_code == 200
    assert client.get(f"/runs/{rid}/download/S1_R2.fastq").status_code == 200
    assert client.get(f"/runs/{rid}/download/demux.log").status_code == 200
    assert client.post(f"/runs/{rid}/groups/S1/release").status_code == 404
    st = client.get(f"/api/runs/{rid}").json()
    assert qc_of(st)["S1"] is None

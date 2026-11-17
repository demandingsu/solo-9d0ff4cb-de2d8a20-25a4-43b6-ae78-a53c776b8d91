"""泳道拆样台:上传 R1/R2/I1 与样本索引 CSV,按样本拆出成对读段。

发布模型:
- 处理中:上传件与半成品输出都在 staging/<token>/,不对外可见。
- 成功:输出目录整体 os.rename 到 published/<运行号>/,同事务写入计数,
  之后状态置为 published。任何一步失败都不会留下半套已发布文件。
- 失败/中断:状态置 failed,暂存与可能的半成品目录被清理;同一运行号可重试。
- 已发布:同一运行号拒绝改写。
"""
import os
import re
import shutil
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .config import db_path, published_dir, staging_dir
from .db import db_conn, init_db
from .demux import demultiplex
from .samples import load_samples

RUN_NUMBER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
STATUS_LABELS = {"processing": "处理中", "published": "已发布", "failed": "失败"}
QC_LABELS = {
    "released": "放行",
    "held": "暂扣",
    "no_data": "无数据",
    "unevaluated": "未评估",  # 旧运行没有质控记录
}
QC_HOLD_PCT = 10  # 任一端低于 Q20 的碱基占比超过该百分比即暂扣
UNDETERMINED = "undetermined"


def qc_verdict(r1_total: int, r1_below: int, r2_total: int, r2_below: int) -> str:
    """空组无数据;任一端低于 Q20 占比 >10% 暂扣;其余放行。

    用整数比较 below*100 > total*QC_HOLD_PCT,避免浮点边界误差。
    """
    if r1_total == 0 and r2_total == 0:
        return "no_data"
    if (r1_below * 100 > r1_total * QC_HOLD_PCT
            or r2_below * 100 > r2_total * QC_HOLD_PCT):
        return "held"
    return "released"


def pct_text(below: int, total: int) -> str:
    return f"{below / total * 100:.2f}%" if total else "—"

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _remove(path) -> None:
    p = Path(path)
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except (FileNotFoundError, NotADirectoryError):
            pass


def recover_interrupted() -> None:
    """启动恢复:凡未走到 published 的运行一律标失败并清掉半成品,
    保证发布区里只存在完整运行的完整文件。"""
    with db_conn(db_path()) as conn:
        stale = conn.execute(
            "SELECT id, run_number, staging_dir FROM runs WHERE status = 'processing'"
        ).fetchall()
        for row in stale:
            conn.execute(
                "UPDATE runs SET status='failed', error=?, updated_at=? WHERE id=?",
                ("进程中断,未完成即停止;可用同一运行号重新提交", utcnow(), row["id"]),
            )
        published = {
            r["run_number"]
            for r in conn.execute("SELECT run_number FROM runs WHERE status='published'")
        }
    for row in stale:
        _remove(row["staging_dir"])
        _remove(published_dir() / row["run_number"])
    if staging_dir().exists():
        for child in staging_dir().iterdir():  # 无主的暂存目录一并清理
            _remove(child)
    if published_dir().exists():
        for child in published_dir().iterdir():
            if child.name not in published:
                _remove(child)
    with db_conn(db_path()) as conn:  # 已发布但文件丢失:不允许提供残缺下载
        for num in published:
            if not (published_dir() / num).is_dir():
                conn.execute(
                    "UPDATE runs SET status='failed', error=?, updated_at=? "
                    "WHERE run_number=? AND status='published'",
                    ("已发布文件缺失,运行标记为失败", utcnow(), num),
                )


def setup() -> None:
    staging_dir().mkdir(parents=True, exist_ok=True)
    published_dir().mkdir(parents=True, exist_ok=True)
    init_db(db_path())
    recover_interrupted()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup()
    yield


app = FastAPI(title="泳道拆样台", lifespan=lifespan)


# ---------------------------------------------------------------- 后台处理

def write_log(path: Path, run, sample_list, stats) -> None:
    lines = [
        f"运行号: {run['run_number']}",
        f"提交时间: {run['created_at']}",
        f"完成时间: {utcnow()}",
        f"输入文件: R1={run['r1_name']} R2={run['r2_name']} "
        f"I1={run['i1_name']} 样本表={run['samples_name']}",
        "样本索引:",
    ]
    lines += [f"  {sid}\t{idx}" for sid, idx in sample_list]
    lines.append(f"总读段对: {stats['total_pairs']}")
    lines.append("分组对数(对数\t其中单碱基纠错):")
    for sid in stats["order"]:
        pairs, corrected = stats["counts"][sid]
        lines.append(f"  {sid}\t{pairs}\t{corrected}")
    lines.append(
        f"质控(Phred+33,低于 Q20 碱基占比,任一端 >{QC_HOLD_PCT}% 暂扣;"
        "暂扣组需质控确认放行后供下载):"
    )
    for sid in stats["order"]:
        r1t, r1b, r2t, r2b = stats["qc"][sid]
        verdict = qc_verdict(r1t, r1b, r2t, r2b)
        if verdict == "no_data":
            lines.append(f"  {sid}\t无数据")
        else:
            lines.append(
                f"  {sid}\tR1={pct_text(r1b, r1t)}\tR2={pct_text(r2b, r2t)}"
                f"\t{QC_LABELS[verdict]}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_run(run_id: int) -> None:
    with db_conn(db_path()) as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if run is None:
        return
    run_number = run["run_number"]
    staging = Path(run["staging_dir"])
    uploads = staging / "uploads"
    out = staging / "out"
    final = published_dir() / run_number
    try:
        sample_list = load_samples(uploads / "samples.csv")
        out.mkdir(parents=True)
        stats = demultiplex(
            str(uploads / "r1.fastq"), str(uploads / "r2.fastq"),
            str(uploads / "i1.fastq"), sample_list, str(out),
        )
        write_log(out / "demux.log", run, sample_list, stats)
        if final.exists():
            raise RuntimeError(f"发布目录已存在: {final}")
        os.rename(out, final)  # 同一数据卷内原子移动:先完整落盘,再发布
        with db_conn(db_path()) as conn:
            updated = conn.execute(
                "UPDATE runs SET status='published', total_pairs=?, published_at=?, "
                "updated_at=? WHERE id=? AND status='processing'",
                (stats["total_pairs"], utcnow(), utcnow(), run_id),
            ).rowcount
            if updated != 1:
                raise RuntimeError("运行状态已变化,放弃发布")
            index_of = dict(sample_list)
            for sid in stats["order"]:
                pairs, corrected = stats["counts"][sid]
                conn.execute(
                    "INSERT INTO sample_counts(run_id, sample_id, index_seq, pairs, "
                    "corrected) VALUES (?,?,?,?,?)",
                    (run_id, sid, index_of.get(sid, ""), pairs, corrected),
                )
                r1t, r1b, r2t, r2b = stats["qc"][sid]
                conn.execute(
                    "INSERT INTO sample_qc(run_id, sample_id, r1_total, r1_below, "
                    "r2_total, r2_below, qc_status) VALUES (?,?,?,?,?,?,?)",
                    (run_id, sid, r1t, r1b, r2t, r2b,
                     qc_verdict(r1t, r1b, r2t, r2b)),
                )
        _remove(staging)
    except Exception as exc:  # 任何失败:标失败 + 清半成品,不发布
        with db_conn(db_path()) as conn:
            conn.execute(
                "UPDATE runs SET status='failed', error=?, updated_at=? WHERE id=?",
                (str(exc), utcnow(), run_id),
            )
        _remove(staging)
        _remove(final)


# ---------------------------------------------------------------- 页面与接口

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if "text/html" in request.headers.get("accept", ""):
        return templates.TemplateResponse(
            request, "error.html",
            {"status": exc.status_code, "detail": exc.detail},
            status_code=exc.status_code,
        )
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    with db_conn(db_path()) as conn:
        runs = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    return templates.TemplateResponse(
        request, "index.html", {"runs": runs, "labels": STATUS_LABELS}
    )


async def save_upload(upload: UploadFile, dest: Path) -> None:
    with open(dest, "wb") as fh:
        while chunk := await upload.read(1 << 20):
            fh.write(chunk)


@app.post("/runs")
async def create_run(
    run_number: str = Form(...),
    r1: UploadFile = File(...),
    r2: UploadFile = File(...),
    i1: UploadFile = File(...),
    samples: UploadFile = File(...),
):
    run_number = run_number.strip()
    if not RUN_NUMBER_RE.fullmatch(run_number):
        raise HTTPException(400, "运行号格式不正确:字母或数字开头,可含 - _ .,最长 64 字符")
    for label, up in (("R1", r1), ("R2", r2), ("I1", i1), ("样本表", samples)):
        if not up.filename:
            raise HTTPException(400, f"请上传{label}文件")

    token = uuid.uuid4().hex
    staging = staging_dir() / token
    uploads = staging / "uploads"
    names = {k: Path(u.filename or "").name for k, u in
             (("r1", r1), ("r2", r2), ("i1", i1), ("samples", samples))}
    try:
        with db_conn(db_path()) as conn:
            row = conn.execute(
                "SELECT id, status FROM runs WHERE run_number=?", (run_number,)
            ).fetchone()
            if row and row["status"] == "published":
                raise HTTPException(409, f"运行 {run_number} 已发布,不可改写")
            if row and row["status"] == "processing":
                raise HTTPException(409, f"运行 {run_number} 正在处理中,请稍后查看")
            if row:  # failed:允许同号重试
                conn.execute("DELETE FROM runs WHERE id=?", (row["id"],))
            cur = conn.execute(
                "INSERT INTO runs(run_number, status, staging_dir, r1_name, r2_name, "
                "i1_name, samples_name, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (run_number, "processing", str(staging), names["r1"], names["r2"],
                 names["i1"], names["samples"], utcnow(), utcnow()),
            )
            run_id = cur.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"运行 {run_number} 已存在,请刷新后查看")

    try:
        uploads.mkdir(parents=True)
        await save_upload(r1, uploads / "r1.fastq")
        await save_upload(r2, uploads / "r2.fastq")
        await save_upload(i1, uploads / "i1.fastq")
        await save_upload(samples, uploads / "samples.csv")
        load_samples(uploads / "samples.csv")  # 先校验样本表,尽早报错
    except HTTPException:
        raise
    except Exception as exc:
        with db_conn(db_path()) as conn:
            conn.execute(
                "UPDATE runs SET status='failed', error=?, updated_at=? WHERE id=?",
                (f"上传或样本表错误: {exc}", utcnow(), run_id),
            )
        _remove(staging)
        raise HTTPException(400, f"上传或样本表错误: {exc}")

    threading.Thread(target=process_run, args=(run_id,), daemon=True).start()
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


def get_run_or_404(run_id: int):
    with db_conn(db_path()) as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if run is None:
        raise HTTPException(404, "运行不存在")
    return run


def get_counts(run_id: int):
    with db_conn(db_path()) as conn:
        return conn.execute(
            "SELECT * FROM sample_counts WHERE run_id=? "
            "ORDER BY sample_id = 'undetermined', sample_id",
            (run_id,),
        ).fetchall()


def get_qc_map(run_id: int):
    """{样本号: 质控记录};旧运行没有记录,各组按未评估处理。"""
    with db_conn(db_path()) as conn:
        rows = conn.execute(
            "SELECT * FROM sample_qc WHERE run_id=?", (run_id,)
        ).fetchall()
    return {r["sample_id"]: r for r in rows}


def group_view(count, qc) -> dict:
    """运行页一组的展示模型:计数、双端占比、质控状态与可否下载。"""
    g = {
        "sample_id": count["sample_id"],
        "index_seq": count["index_seq"],
        "pairs": count["pairs"],
        "corrected": count["corrected"],
        "r1": f"{count['sample_id']}_R1.fastq",
        "r2": f"{count['sample_id']}_R2.fastq",
    }
    if qc is None:  # 旧运行:无质量数据,保持原下载行为
        g.update(qc_status="unevaluated", qc_label=QC_LABELS["unevaluated"],
                 r1_pct="—", r2_pct="—", confirmed=False, downloadable=True)
        return g
    status = qc["qc_status"]
    confirmed = bool(qc["released_by_qc"])
    g.update(
        qc_status=status,
        qc_label=QC_LABELS[status],
        r1_pct=pct_text(qc["r1_below"], qc["r1_total"]),
        r2_pct=pct_text(qc["r2_below"], qc["r2_total"]),
        confirmed=confirmed,
        downloadable=(status == "released"
                      or (status == "held" and confirmed)),
    )
    return g


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: int):
    run = get_run_or_404(run_id)
    qc_map = get_qc_map(run_id) if run["status"] == "published" else {}
    groups = [
        group_view(c, qc_map.get(c["sample_id"])) for c in get_counts(run_id)
    ]
    return templates.TemplateResponse(
        request, "run.html",
        {"run": run, "groups": groups, "labels": STATUS_LABELS,
         "qc_hold_pct": QC_HOLD_PCT},
    )


@app.post("/runs/{run_id}/groups/{sample_id}/release")
def confirm_release(run_id: int, sample_id: str):
    """质控员确认放行暂扣组;幂等,确认后重启仍有效(落库即持久)。"""
    run = get_run_or_404(run_id)
    if run["status"] != "published":
        raise HTTPException(409, "运行未发布,不能确认放行")
    with db_conn(db_path()) as conn:
        row = conn.execute(
            "SELECT qc_status, released_by_qc FROM sample_qc "
            "WHERE run_id=? AND sample_id=?",
            (run_id, sample_id),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "该组没有质控记录,无需确认")
        if row["qc_status"] != "held":
            raise HTTPException(409, "该组不在暂扣状态,无需确认")
        conn.execute(
            "UPDATE sample_qc SET released_by_qc=1, "
            "confirmed_at=COALESCE(confirmed_at, ?) "
            "WHERE run_id=? AND sample_id=? AND qc_status='held'",
            (utcnow(), run_id, sample_id),
        )
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@app.get("/api/runs/{run_id}")
def run_status(run_id: int):
    run = get_run_or_404(run_id)
    counts = get_counts(run_id)
    qc_map = get_qc_map(run_id)
    return {
        "run_number": run["run_number"],
        "status": run["status"],
        "error": run["error"],
        "total_pairs": run["total_pairs"],
        "counts": [
            {"sample_id": c["sample_id"], "index_seq": c["index_seq"],
             "pairs": c["pairs"], "corrected": c["corrected"],
             "qc": qc_json(qc_map.get(c["sample_id"]))}
            for c in counts
        ],
    }


def qc_json(qc):
    if qc is None:
        return None  # 旧运行:未评估
    return {
        "r1_total": qc["r1_total"], "r1_below": qc["r1_below"],
        "r2_total": qc["r2_total"], "r2_below": qc["r2_below"],
        "status": qc["qc_status"],
        "confirmed": bool(qc["released_by_qc"]),
    }


@app.get("/runs/{run_id}/download/{filename}")
def download(run_id: int, filename: str):
    run = get_run_or_404(run_id)
    if run["status"] != "published":
        raise HTTPException(409, "运行未发布,没有完整文件可下载")
    file_to_sid = {}
    for c in get_counts(run_id):
        file_to_sid[f"{c['sample_id']}_R1.fastq"] = c["sample_id"]
        file_to_sid[f"{c['sample_id']}_R2.fastq"] = c["sample_id"]
    if filename != "demux.log" and filename not in file_to_sid:
        raise HTTPException(404, "文件不存在")
    sid = file_to_sid.get(filename)
    if sid is not None:  # 按已提交的组质控状态供数;无记录的旧运行保持原行为
        qc = get_qc_map(run_id).get(sid)
        if qc is not None:
            if qc["qc_status"] == "no_data":
                raise HTTPException(409, "该组无数据,没有可下载文件")
            if qc["qc_status"] == "held" and not qc["released_by_qc"]:
                raise HTTPException(
                    409, "该组低质量暂扣中,需质控确认放行后才能下载")
    path = published_dir() / run["run_number"] / filename
    if not path.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(path, filename=filename, media_type="application/octet-stream")

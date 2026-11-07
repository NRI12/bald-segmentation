import shutil
from pathlib import Path

from baldseg.data import download

SOURCES = {
    ("fifa-eg5yw", "hairloss-segmentation-lf62l", 1, "yolov8"): {
        "data_397_png.rf.da94c7a275ba1be7000c4076eab88d8e.jpg": "seg_a.jpg",
        "data_455_png.rf.c6f35e59408a1b75b35af58ae1e7693c.jpg": "seg_b.jpg",
    },
    ("uze", "hair-loss-nq8hh", 1, "folder"): {
        "23-Top-Down_jpg.rf.45eeaf361abf95397e963c056ee3c25e.jpg": "norwood2_23.jpg",
        "55-Top-Down_jpg.rf.568e4282c7095ee1566b3c6aae0a14b6.jpg": "norwood3_55.jpg",
        "31-Top-Down_jpg.rf.2665c0b7a4a3c07905c858b59d8afee2.jpg": "norwood4_31.jpg",
        "11-Top-Down_jpg.rf.aa741232c58ec0cbdd9172201c812854.jpg": "norwood5_11.jpg",
        "18-Top-Down_jpg.rf.4b492d220f19edb0446c77912fbecf35.jpg": "norwood6_18.jpg",
        "10-Top-Down_jpg.rf.6e17397fc6bdf7abf31fa560e2949156.jpg": "norwood7_10.jpg",
    },
}


def main(out=Path("examples")):
    out.mkdir(exist_ok=True)
    for (workspace, project, version, fmt), files in SOURCES.items():
        raw = Path(download(workspace, project, version, fmt, Path("raw_examples") / project))
        for src, dst in files.items():
            shutil.copy(next(raw.rglob(src)), out / dst)
    print(sorted(p.name for p in out.glob("*.jpg")))


if __name__ == "__main__":
    main()

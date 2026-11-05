import shutil
from pathlib import Path

from baldseg.data import download

SOURCES = {
    ("fifa-eg5yw", "hairloss-segmentation-lf62l", 1, "yolov8"): {
        "data_299_png.rf.9fee1564b45c99202925fcf18f077335.jpg": "seg1_Top-Down.jpg",
        "data_397_png.rf.da94c7a275ba1be7000c4076eab88d8e.jpg": "seg2_Top-Down.jpg",
        "data_455_png.rf.c6f35e59408a1b75b35af58ae1e7693c.jpg": "seg3_Top-Down.jpg",
        "data_208_png.rf.1246233bdbca22cccc84e430e4f46e58.jpg": "seg4_Top-Down.jpg",
    },
    ("uze", "hair-loss-nq8hh", 1, "folder"): {
        "23-Front_jpg.rf.c83ab3bc1640bf4379b1ee8a1566ec51.jpg": "norwood2_Front.jpg",
        "35-Top-Down_jpg.rf.1888bc01875c83d34c1a2f8a091b63d2.jpg": "norwood4_Top-Down.jpg",
        "11-Front_jpg.rf.b8d3990880494d68bb35cce2fa0be11c.jpg": "norwood5_Front.jpg",
        "1-Front_jpg.rf.7bb2147ccbc21d266846c229c8d87d8e.jpg": "norwood7_Front.jpg",
    },
}


def main(out=Path("examples")):
    out.mkdir(exist_ok=True)
    for (workspace, project, version, fmt), files in SOURCES.items():
        raw = Path(download(workspace, project, version, fmt, Path("raw_examples") / project))
        for src, dst in files.items():
            shutil.copy(next(raw.rglob(src)), out / dst)
    print(sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()

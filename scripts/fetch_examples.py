import shutil
from pathlib import Path

from baldseg.data import download

SOURCES = {
    ("fifa-eg5yw", "hairloss-segmentation-lf62l", 1, "yolov8"): {
        "data_397_png.rf.da94c7a275ba1be7000c4076eab88d8e.jpg": "seg_a/Top-Down.jpg",
        "data_455_png.rf.c6f35e59408a1b75b35af58ae1e7693c.jpg": "seg_b/Top-Down.jpg",
    },
    ("uze", "hair-loss-nq8hh", 1, "folder"): {
        "23-Front_jpg.rf.38b5e37c68764a35f68956236edbbf75.jpg": "norwood2_23/Front.jpg",
        "23-Top-Down_jpg.rf.45eeaf361abf95397e963c056ee3c25e.jpg": "norwood2_23/Top-Down.jpg",
        "23-Back_jpg.rf.f5490e275ba15e6a18935f4634c5f0cc.jpg": "norwood2_23/Back.jpg",
        "23-Left_jpg.rf.16d975d02596e3cd4df79c32393e6ae9.jpg": "norwood2_23/Left.jpg",
        "23-Right_jpg.rf.bf16585440f16e893e95652a3a0349af.jpg": "norwood2_23/Right.jpg",
        "55-Front_jpg.rf.8e33204ae51619dbc14b2c05374a5816.jpg": "norwood3_55/Front.jpg",
        "55-Top-Down_jpg.rf.568e4282c7095ee1566b3c6aae0a14b6.jpg": "norwood3_55/Top-Down.jpg",
        "55-Back_jpg.rf.d9d8c2d05c502100062a22d6b06b67ac.jpg": "norwood3_55/Back.jpg",
        "55-Left_jpg.rf.0e2f86ace456873ba2230c1db0a2ce0b.jpg": "norwood3_55/Left.jpg",
        "55-Right_jpg.rf.84e1ebb34ab1503e3a065c159dda5f0f.jpg": "norwood3_55/Right.jpg",
        "31-Front_jpg.rf.1d4711bb82f75279ee2a40c486f28384.jpg": "norwood4_31/Front.jpg",
        "31-Top-Down_jpg.rf.2665c0b7a4a3c07905c858b59d8afee2.jpg": "norwood4_31/Top-Down.jpg",
        "31-Back_jpg.rf.1e03221e8501fb77ca144fc3ee86b2c7.jpg": "norwood4_31/Back.jpg",
        "31-Left_jpg.rf.5ef06b8de4acd31da7f48d7473c0d95d.jpg": "norwood4_31/Left.jpg",
        "31-Right_jpg.rf.0e9847705e903a9771d991363c803af0.jpg": "norwood4_31/Right.jpg",
        "11-Front_jpg.rf.b8d3990880494d68bb35cce2fa0be11c.jpg": "norwood5_11/Front.jpg",
        "11-Top-Down_jpg.rf.aa741232c58ec0cbdd9172201c812854.jpg": "norwood5_11/Top-Down.jpg",
        "11-Back_jpg.rf.3c45c46cd15b9b0e001a74a7dd35f7c4.jpg": "norwood5_11/Back.jpg",
        "11-Left_jpg.rf.51d87e915efbc8cef264f80b649c34c2.jpg": "norwood5_11/Left.jpg",
        "11-Right_jpg.rf.0bc760f11310dd6860d241e15f82629e.jpg": "norwood5_11/Right.jpg",
        "18-Front_jpg.rf.3838c669eeac63c9b12e76cdb11f6cc3.jpg": "norwood6_18/Front.jpg",
        "18-Top-Down_jpg.rf.4b492d220f19edb0446c77912fbecf35.jpg": "norwood6_18/Top-Down.jpg",
        "18-Back_jpg.rf.a78559892c3cb59c53b7080b48b6b754.jpg": "norwood6_18/Back.jpg",
        "18-Left_jpg.rf.0afa28f7af1b7117782511c19fd6ec27.jpg": "norwood6_18/Left.jpg",
        "18-Right_jpg.rf.23fa9c9391c6a7682a8b2a6230467332.jpg": "norwood6_18/Right.jpg",
        "10-Front_jpg.rf.fcd10ddf79b850968030d4b93c75f250.jpg": "norwood7_10/Front.jpg",
        "10-Top-Down_jpg.rf.6e17397fc6bdf7abf31fa560e2949156.jpg": "norwood7_10/Top-Down.jpg",
        "10-Back_jpg.rf.25487e8aaba1060d4ec02f33cb4319cd.jpg": "norwood7_10/Back.jpg",
        "10-Left_jpg.rf.2c18ec70ca6c82e208d237384350c620.jpg": "norwood7_10/Left.jpg",
        "10-Right_jpg.rf.6ac40bf0e215bf44e8c73488d3140fde.jpg": "norwood7_10/Right.jpg",
    },
}


def main(out=Path("examples")):
    for (workspace, project, version, fmt), files in SOURCES.items():
        raw = Path(download(workspace, project, version, fmt, Path("raw_examples") / project))
        for src, dst in files.items():
            (out / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(next(raw.rglob(src)), out / dst)
    print(sorted(str(p.relative_to(out)) for p in out.rglob("*.jpg")))


if __name__ == "__main__":
    main()

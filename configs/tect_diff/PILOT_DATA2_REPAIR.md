# CASIA2 pilot DATA2: explicit two-pair quarantine

Pilot A (`TECT-PILOT-CASIA2-GN8-R512-S42-20260916-A`, source `adf3102c329a5bc019355f14cb142c6de916acdc`) stopped during read-only data preparation. Its controller published a FAILED/HOLD report and released all resource locks. No GPU worker, model initialization, training step or checkpoint existed.

The audit inspected 12,614 input pairs and accepted 12,612. The two rejected masks are PIL `LA` (luminance plus alpha), 600×600; the existing training decoder treats their two channels as a color-channel consensus check and rejects disagreement. Luminance gives 57,618 positive pixels; alpha gives 359,799, with 302,189 differing threshold decisions. No model output was used to identify these files. Alpha semantics have not been registered for this training contract, so this bounded revision excludes both exact pairs instead of changing the decoder or the data.

| CASIA2 stem | Image SHA-256 |
| --- | --- |
| Tp_D_NRN_M_N_nat10134_nat00095_11912 | `0432402f0e1adaa07607304b5d3db636c08176bf74c65245775961afd88fac6f` |
| Tp_D_NRN_M_N_nat10134_nat10124_11913 | `a2510909a949940ed64f61abd77ca88ef168814be92d90cdccc33d825c2eaf2a` |

Both mask files have SHA-256 `bc93a054bd80d87ccb5cbf2b059edb014690da7a6d878f9171f78e864989abe4`. The configuration records these IDs, both file hashes, and the reason. An unknown or duplicate ID, changed file hash, missing reason, or non-training record fails closed. Other unresolved semantic errors still stop preparation. All original images and masks remain untouched; manifests record the two exclusions and the original population count.

New run: `TECT-PILOT-CASIA2-GN8-R512-S42-20260916-B`; protocol `TECT-PILOT-CASIA2-GN8-R512-S42-DATA2-V1`; config [pilot_casia2_gn8_data2_r512_s42.json](pilot_casia2_gn8_data2_r512_s42.json). A stays held. B retains fresh seed42/ImageNet MAIN, GN8, the original frozen reference/calibration, micro6/global12, 2,048 balanced training images, 128+128 quick test images, 64 authentic probes, ten-epoch/four-hour limits and every preregistered promotion threshold. It does not resume A or add a scientific comparison arm.

The user's 2026-09-16 authorization covers bounded supervised repairs without repeat approval. This is a versioned data preparation repair; it makes no model-effectiveness claim. All subsequent scores remain `selection_protocol=test_selected` development feedback.

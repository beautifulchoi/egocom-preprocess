# EgoCom Preprocessing Procedure

This file is a compact overview of the preprocessing order. Detailed stage-level explanations are kept in the existing HTML files linked below.

## Progress Diagram

```mermaid
flowchart TD
    raw[Raw EgoCom 1-minute chunks<br/>video, frames, audio]
    depth[DA3 depth and intrinsics<br/>src/step_01_extract_depth.py]
    mask[SAM3 person masks<br/>src/step_02_extract_mask.py]
    refine[Depth-filtered person masks<br/>src/step_03_filter_mask.py]
    face[Face / StyleID embeddings<br/>src/step_04_extract_person_embeding.py]
    map[Chunk-local person mapping<br/>src/step_05_map_person_face_embedding.py]
    report[Mapping reports<br/>src/step_06_report_person_face_mapping.py]
    remap[All-chunk remapping<br/>src/step_07_remap_person_face_all_chunks.py]
    merged_report[Merged segment reports<br/>src/step_08_report_merged_segments.py]
    final_mask[Final per-person mask JPGs<br/>src/step_09_extract_final_person_masks.py]
    lift[Person depth lift<br/>src/step_10_extract_person_depth_lift.py]
    clip[Masked person CLIP features<br/>src/step_11_extract_person_visual_clip.py]
    mclip[Mask-pooled CLIP patch features<br/>src/step_12_extract_person_masked_clip.py]
    mda3[Mask-pooled DA3 features<br/>src/step_13_extract_person_masked_da3.py]
    pe[Mask-pooled positional encoding<br/>src/step_14_extract_pe.py]
    text[InternVL2 spatial text<br/>src/step_15_extract_person_internvl2_text.py]
    t5[T5 spatial text features<br/>src/step_16_extract_person_t5_text_features.py]
    manifest[Windowed multimodal manifest<br/>src/step_17_build_egocom_window_manifest.py]

    raw --> depth
    raw --> mask
    depth --> refine
    mask --> refine
    refine --> face
    face --> map
    refine --> map
    map --> report
    map --> remap
    remap --> merged_report
    remap --> final_mask
    final_mask --> lift
    final_mask --> clip
    final_mask --> mclip
    final_mask --> mda3
    final_mask --> pe
    final_mask --> text
    remap --> lift
    remap --> clip
    remap --> mclip
    remap --> mda3
    remap --> pe
    remap --> text
    depth --> lift
    text --> t5
    lift --> manifest
    clip --> manifest
    pe --> manifest
    t5 --> manifest
    raw --> manifest
```

Use the table below to open each detailed HTML reference.

## Stage Links

| Order | Stage | Detailed HTML reference | Main script | Primary output |
| --- | --- | --- | --- | --- |
| 0 | Source chunks | Input prerequisite | Existing dataset layout | `/home/prj/data/egocom_holdout/1min/{split}` and original audio |
| 1 | DA3 depth and intrinsics | Used by [Depth-Based Mask Refinement](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_03_filter_mask_explanation.html) and [Person Depth Lift](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_10_person_depth_lift_explanation.html) | `src/step_01_extract_depth.py` | `{split}/da3/monocular`, `{split}/da3/nested` |
| 2 | SAM3 person masks | Input to [Depth-Based Mask Refinement](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_03_filter_mask_explanation.html) | `src/step_02_extract_mask.py` | `{split}/person_mask/{video}` |
| 3 | Depth-filter masks | [Depth-Based Mask Refinement](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_03_filter_mask_explanation.html) | `src/step_03_filter_mask.py` | `{split}/refined_mask/{video}` |
| 4 | Extract face embeddings | [Person Face Embedding Extraction](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_04_extract_person_embeding_explanation.html) | `src/step_04_extract_person_embeding.py` | Per-video face embedding outputs from refined masks |
| 5 | Map segment IDs to scene person IDs | [Face-Embedding Person Mapping](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_05_map_person_face_embedding_explanation.html) | `src/step_05_map_person_face_embedding.py` | `{split}/person_face_mapping/{scene}` |
| 6 | Inspect mapping outputs | [Person Mapping Outputs Guide](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_06_person_face_report_outputs_guide.html), [Current Mapping Report](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_06_person_face_mapping_report.html) | `src/step_06_report_person_face_mapping.py` | Mapping report HTML/JSON summaries |
| 7 | Remap all chunks from selected prototypes | [All-Chunk Person Remapping](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_07_remap_person_face_all_chunks_explanation.html) | `src/step_07_remap_person_face_all_chunks.py` | `remap_all_chunks.json`, remap summaries |
| 8 | Inspect merged/remapped cases | [Person Mapping Outputs Guide](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_06_person_face_report_outputs_guide.html) | `src/step_08_report_merged_segments.py` | Merged segment report files |
| 9 | Save final per-person masks | [Final Person Mask Extraction](docs/step/step_09_final_person_mask_extraction_explanation.html) | `src/step_09_extract_final_person_masks.py` | `{split}/final_mask/{scene}/chunk_XXXX/{video}/person_{id}` |
| 10 | Lift mapped persons into depth summaries | [Person Depth Lift](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_10_person_depth_lift_explanation.html) | `src/step_10_extract_person_depth_lift.py` | `{split}/person_depth_lift/{scene}/person_{id}` |
| 11 | Extract masked visual CLIP features | [Masked Person CLIP Features](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_11_person_visual_clip_explanation.html) | `src/step_11_extract_person_visual_clip.py` | `{split}/person_visual_clip_features/{scene}/person_{id}` |
| 12 | Extract mask-pooled CLIP features | [Mask-Pooled CLIP/DA3 Features](docs/step/step_12_person_masked_pooling_features_explanation.html) | `src/step_12_extract_person_masked_clip.py` | `{split}/person_masked_clip_features/{scene}/person_{id}` |
| 13 | Extract mask-pooled DA3 features | [Mask-Pooled CLIP/DA3 Features](docs/step/step_12_person_masked_pooling_features_explanation.html) | `src/step_13_extract_person_masked_da3.py` | `{split}/person_masked_da3_features/{scene}/person_{id}` |
| 14 | Extract positional encoding features | [Mask-Pooled Positional Encoding](docs/step/step_14_extract_pe_explanation.html) | `src/step_14_extract_pe.py` | `{split}/person_pe_features/{scene}/person_{id}` |
| 15 | Generate spatial text | [Person Spatial Text Features](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_15_person_spatial_text_features_explanation.html) | `src/step_15_extract_person_internvl2_text.py` | `{split}/person_spatial_internvl2_text/{scene}/person_{id}` |
| 16 | Encode spatial text with T5 | [Person Spatial Text Features](https://htmlpreview.github.io/?https://github.com/beautifulchoi/egocom-preprocess/blob/main/docs/step/step_15_person_spatial_text_features_explanation.html) | `src/step_16_extract_person_t5_text_features.py` | `{split}/person_spatial_t5_features/{scene}/person_{id}` |
| 17 | Build windowed multimodal manifest | Final packaging step | `src/step_17_build_egocom_window_manifest.py` | `{output_tag}/{split}/jsonl/manifest.jsonl`, audio windows, depth rays, CLIP, PE, and T5 features |

## Dependency Notes

- Step 9 materializes final per-person masks after all-chunk remapping is available.
- Steps 10 through 15 use the same remapped person tracks for downstream geometry, visual, positional, and text features.
- The window manifest currently consumes person depth-lift outputs, person visual CLIP features, mask-pooled PE features, T5 features, and original audio.
- Spatial text/T5 and PE features are produced as parallel person-level feature branches.

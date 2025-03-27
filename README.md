# Enhanced Multimodal Hate Video Detection via Channel-wise and Modality-wise Fusion.

# Dataset Processing Instructions

## Store Video Names and Labels
- Use the provided `.csv` file from the HateMM dataset to extract video names and their corresponding labels.
- Save this information in the `final_allNewData.pkl` file.
- **Script**: `final_allNewData.py`

---

## Dataset Splitting
The dataset is divided into **5 folds** to ensure consistent training/testing splits:
- **Training set**: 70%
- **Test set**: 20%
- **Validation set**: 10%
- Split details are stored in `allFoldDetails.pkl`.
- **Script**: `alldetail.py`

---

## Extract Audio from Videos
- Extract audio data from video files.
- **Script**: `video_to_audio.py`

---

## Extract Text from Audio
- Convert extracted audio into text data.
- **Script**: `audio_to_text.py`

---

## Baseline Model
- Uses **GPT-3.5** for text processing.
- API calls are implemented in:  
  **Script**: `GPT_Text_baseline.py`

---

## CMFusion Model
Run multimodal fusion experiments using:
```bash
python CMFusion.py

## Acknowledgements
This study builds upon the baseline code architecture released by the HateMM team. We gratefully acknowledge their contributions in providing the high-quality multimodal hate speech detection dataset and baseline model implementation.

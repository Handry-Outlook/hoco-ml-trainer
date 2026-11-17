# hoco-ml-trainer

Daily training for **AI HOCO**, Handry Outlook's machine-learned HOCO lightning forecast.

- `train_hoco.py`: ingests verified HOCO runs, learns per-model weights, weather context and calibration, and publishes them to Firebase.
- `hoco_ml.py`, `hoco_context.py`: shared with the HOCO backend. Keep these identical to the copies in the backend repos.

The workflow is started by Firebase (daily) or from the AI HOCO Lab. Credentials come from the `FIREBASE_SERVICE_ACCOUNT` repository secret; none are stored in this repo.

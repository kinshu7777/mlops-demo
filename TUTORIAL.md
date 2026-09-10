# Tutorial: Running This Project Step by Step

This is the plain-language walkthrough. If you're comfortable with
Python already, the [README](README.md) covers the same ground more
technically. If anything here is unclear, that's this guide's fault,
not yours — follow along one command at a time.

---

## Before you start

You need Python installed. Check with:

```bash
python3 --version
```

If that shows a version number, you're ready.

---

## The fastest way to get started

If you just want to see it running without reading each step below:

- **Windows:** double-click `start.bat`
- **Mac/Linux:** double-click `start.sh` (or run `./start.sh` in a terminal)

Either one automatically installs nothing extra, but does set up the
practice data and a trained model if they're missing, then starts the
server for you - all in one step. Then visit `http://127.0.0.1:8000`.
If anything about that felt unclear, the steps below explain each
piece individually.

## Step 0: Install the ingredients

```bash
pip install -r requirements.txt
```

`requirements.txt` is a shopping list of Python packages this project
needs. This command installs all of them at once.

---

## Step 1: Create some practice data

```bash
python3 control.py generate-data
```

This invents a pretend delivery company's data and saves it as two
files: `old_deliveries.csv` (last year's data, used for training) and
`new_deliveries.csv` (this month's data, built to look slightly
different on purpose, so there's something to detect later).

---

## Step 2: Teach the computer to predict delivery time

```bash
python3 control.py train
```

The computer looks at 2,000 past deliveries and learns the pattern
between (distance, stops, rush hour, experience) and (delivery time).
It prints how accurate the result is, and saves the model as a
versioned file inside `models/`.

---

## Step 3: Let other programs use your model

```bash
python3 control.py serve
```

Open your browser to `http://127.0.0.1:8000/docs` — an interactive
page where you can click "Try it out" on `/predict`, fill in some
numbers, and get a prediction back instantly. This is what "deploying
a model" means: without this step, the model is just a file sitting
uselessly on your computer.

Leave this running, or `Ctrl+C` to stop it.

---

## Step 4: Check if the world has changed ("drift")

Open a second terminal window and run:

```bash
python3 control.py check-drift
```

This compares this month's data to last year's and asks, for each
column: "does this still look like the same pattern?" You should see
some columns marked `DRIFTED` — expected, since the practice data was
built with a deliberate shift.

**Why this matters:** a model's accuracy quietly rots as the real
world changes. This step is what lets you notice.

---

## Step 5: See it all on one screen

```bash
python3 control.py dashboard
```

Opens a dashboard showing every model version's accuracy and the
latest drift results — the piece you'd screen-share in an interview.

---

## Step 6: The one-command "do the right thing" pipeline

```bash
python3 control.py pipeline
```

Checks for drift, and retrains automatically **only if** drift was
actually found — instead of blindly retraining every time.

---

## Step 7: Run the test suite

```bash
python3 control.py test
```

Runs automated checks confirming training and drift detection behave
correctly, without you having to manually verify by eye.

---

## Step 8 (optional): Automate it with GitHub Actions

`.github/workflows/retrain.yml` tells GitHub to run the drift check
daily and retrain automatically if needed — once you push this
project to a GitHub repository, no human has to remember to run it.

---

## Step 9 (optional): Package it with Docker

```bash
docker build -t delivery-predictor .
docker run -p 8000:8000 delivery-predictor
```

Packages the entire app so it runs identically on any computer,
including a company's servers.

---

## All control.py commands, in one place

```
python3 control.py generate-data      # make practice data
python3 control.py train              # train a new model version
python3 control.py check-drift        # check if data has shifted
python3 control.py predict --distance_km 5 --num_stops 2 --is_rush_hour 1 --driver_experience 3
python3 control.py status             # quick summary: model + drift state
python3 control.py pipeline           # check drift, retrain only if needed
python3 control.py start              # first-time setup + start, all in one
python3 control.py serve              # start the API
python3 control.py dashboard          # start the dashboard
python3 control.py test               # run the test suite
python3 control.py config             # see current settings
python3 control.py --help             # see all commands
```

**Where to change settings:** open `config.json`. Every script reads
from that same file, so a change there applies everywhere at once.

## What to say about this project in an interview

> "I built an end-to-end pipeline that trains a model, serves it
> through an API, and monitors incoming data for drift so it knows
> when to retrain itself. I automated the retraining with GitHub
> Actions, containerized the API with Docker, and wrote tests to
> verify the pipeline's correctness."

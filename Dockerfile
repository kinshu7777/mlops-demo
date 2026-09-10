# STEP 7: Package everything so it runs the same way on any computer.
#
# Think of this like a recipe card: "start with a plain Python kitchen,
# add these ingredients (requirements.txt), copy in the food (our code),
# and here's how to serve it (the last line)."

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

import os

from constrained_generation.service import create_app

# CG_DATA_DIR enables the persistent request ledger (restart recovery);
# defaults to a local .cgdata directory.
app = create_app(data_dir=os.environ.get("CG_DATA_DIR", ".cgdata"))


@app.get("/")
def read_root():
    return {"Hello": "World"}

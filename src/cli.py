import typer
from . import train, predict

app = typer.Typer()
app.add_typer(train.app, name="train-model")
app.add_typer(predict.app, name="predict-data")

if __name__ == "__main__":
    app()

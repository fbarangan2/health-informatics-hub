# demo
from etl.Bronze.ingest_ph_population import ingest_population
from etl.Silver.transform_population import transform_population
from etl.Gold.build_population_dataset import build_population_dataset

def run():
    ingest_population()
    transform_population()
    build_population_dataset()

if __name__ == "__main__":
    run()

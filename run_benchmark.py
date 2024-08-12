import metaformer_baselines
from timm.models import create_model
from misc_utils.benchmark import main


benchmark_list = {
  # "caformer_s18": 224,
  # "featcaformer_s18": 224,
  # "featpartialcaformer_s18": 224,
  # "featpartialcaformer_s14": 224,
  "featpartialcaformer_s12": 224,
}

def get_benchmark():
    for model_name in benchmark_list:
        print(f"Performance measure for {model_name}")
        model = create_model(
                    model_name)
        main(model)

if __name__ == '__main__':
    get_benchmark()
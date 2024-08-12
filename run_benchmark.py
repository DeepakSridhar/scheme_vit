import schemeformer
from timm.models import create_model
from misc_utils.benchmark import main

benchmark_list = {
    "schemeformer_ppaa_s12_224": 224,
    "schemeformer_ppaa_s36_224": 224,
    "grouporgsemlpformer_ppaa_s12_224": 224,
    # "shufflegrouporgattnmlpformer_ppaa_s12_224": 224,
    # "groupliteconvmlpformer_ppaa_s12_224": 224,
    # "groupliteattnmlpformer_ppaa_s12_224": 224,

}

def get_benchmark():
    for model_name in benchmark_list:
        print(f"Performance measure for {model_name}")
        model = create_model(
                    model_name)
        main(model)

if __name__ == '__main__':
    get_benchmark()
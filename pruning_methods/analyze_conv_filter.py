from pruning_methods.statistical_method.statistical_method_univariate import *
from pruning_methods.statistical_method.statistical_method_multivariate import *


def analyze_conv_filter(conv_weights, conv_name, plot=True):
    print("\t-> ✅Analyzing univariate methods")
    drop_list_uni = analyze_univariate(conv_weights, conv_name, plot=True)
    print("\t-> ✅Analyzing multivariate methods")
    drop_list_multi = analyze_multivariate(conv_weights, conv_name, plot=True)
    # Combine (unique sorted)
    combined_drop = sorted(set(drop_list_uni + drop_list_multi))
    return combined_drop
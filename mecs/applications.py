import scipy.stats as stats
from constants import *

app_info={
    SPEECH_RECOGNITION : {'workload':10435,
        'popularity': 0.25,
        'min_bits':80000*BYTE,
        'max_bits':800000*BYTE
    },NLP : {'workload':25346,
        'popularity': 0.1,
        'min_bits':300*BYTE,
        'max_bits':1400*BYTE
    },FACE_RECOGNITION : {'workload':45043,
        'popularity': 0.1,
        'min_bits':300000*BYTE,
        'max_bits':30000000*BYTE
    },SEARCH_REQ : {'workload':8405,
        'popularity': 0.15,
        'min_bits':400,
        'max_bits':700*BYTE
    },LANGUAGE_TRANSLATION : {'workload':34252,
        'popularity': 0.1,
        'min_bits':300*BYTE,
        'max_bits':4000*BYTE
    },PROC_3D_GAME : {'workload':54633,
        'popularity': 0.1,
        'min_bits':100000*BYTE,
        'max_bits':3000000*BYTE
    },VR : {'workload':40305,
        'popularity': 0.1,
        'min_bits':100000*BYTE,
        'max_bits':3000000*BYTE
    },AR : {'workload':34532,
        'popularity': 0.1,
        'min_bits':100000*BYTE,
        'max_bits':3000000*BYTE
    }
}

def app_type_list():
    return app_info.keys()

def app_type_pop():
    # result =[]
    # for i in list(app_info.keys()):
    #     import pdb; pdb.set_trace()
    #     result.append([i, app_info[i]['popularity']])
    # return result
    return [(i, app_info[i]['popularity']) for i in list(app_info.keys())]

def arrival_bits(app_type, dist = 'normal'):
    min_bits = app_info[app_type]['min_bits']
    max_bits = app_info[app_type]['max_bits']
    mu = (min_bits+max_bits)/2
    sigma = (max_bits-min_bits)/4
    if dist=='normal':
        return int(stats.truncnorm.rvs((min_bits-mu)/sigma, (max_bits-mu)/sigma, loc=mu, scale=sigma))
    else:
        return 1
# import pdb; pdb.set_trace()

SPEECH_RECOGNITION = 1
NLP = 2
FACE_RECOGNITION = 3
SEARCH_REQ = 4
LANGUAGE_TRANSLATION = 5
PROC_3D_GAME = 6
VR = 7
AR = 8

app_info={
    1 : {'workload':10435,
        'popularity': 0.25,
        'min_bits':80000,
        'max_bits':800000
    },2 : {'workload':25346,
        'popularity': 0.1,
        'min_bits':300,
        'max_bits':1400
    },3 : {'workload':45043,
        'popularity': 0.1,
        'min_bits':300000,
        'max_bits':30000000
    },4 : {'workload':8405,
        'popularity': 0.15,
        'min_bits':400,
        'max_bits':700
    },5 : {'workload':34252,
        'popularity': 0.1,
        'min_bits':300,
        'max_bits':4000
    },6 : {'workload':54633,
        'popularity': 0.1,
        'min_bits':100000,
        'max_bits':3000000
    },7 : {'workload':40305,
        'popularity': 0.1,
        'min_bits':100000,
        'max_bits':3000000
    },8 : {'workload':34532,
        'popularity': 0.1,
        'min_bits':100000,
        'max_bits':3000000
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
# import pdb; pdb.set_trace()

import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('X', type=int, nargs='+')
parser.add_argument('--length', default = 2000, type=int)

'''
    SPEECH_RECOGNITION : {'workload':10435,
        'popularity': 5,
        'min_bits':40000*BYTE,
        'max_bits':300000*BYTE
    },NLP : {'workload':25346,
        'popularity': 8,
        'min_bits':4000*BYTE,
        'max_bits':100000*BYTE
    },FACE_RECOGNITION : {'workload':45043,
        'popularity': 4,
        'min_bits':10000*BYTE,
        'max_bits':100000*BYTE
    }
'''

dir_names =["ppo_fixed_len2019-09-16 15:00:57.522233","ppo_fixed_len_under1ver2019-10-15 14:12:57.392339","ppo_fixed_len_under1ver2019-10-16 12:10:20.290043", "ppo_fixed_len_under1ver2019-10-16 12:12:04.333238", "ppo_fixed_under1dummy2019-10-17 16:30:23.506688",
"ppo_fixed_under1dummy2019-10-17 16:31:24.518229","ppo_fixed_under1dummy2019-10-18 13:56:40.239857", "ppo_fixed_under1dummy2019-10-18 13:56:48.124068","ppo_fixed_under1dummy2019-10-20 14:58:09.686552","ppo_fixed_under1dummy_newnetwork2019-10-31 19:00:13.652069", "ppo_fixed_under1dummy_newnetwork2019-11-01 13:45:35.362800",
"ppo_fixed_under1dummy_newnetwork2019-11-01 14:10:29.094330","ppo_fixed_under1dummy_newnetwork2019-11-04 10:02:16.187410","ppo_fixed_under1dummy_newnetwork2019-11-04 15:35:57.245712","ppo_fixed_under1dummy_newnetwork2019-11-09 15:28:04.563948","ppo_fixed_under1dummy_newnetwork2019-11-11 13:33:54.212304", "ppo_fixed_under1dummy2019-10-20 14:58:09.686552"]
# final 논문 그림, 처음 1보다 작게 시도한 것, 그다음 것 두개, softmax dummy로 두개



args = parser.parse_args()
graph_list = args.X
length = args.length
for graph in graph_list:
    dir_name = "results/" + dir_names[graph]
    a = np.load("{}/eval_results/eval_empty_reward.npy".format(dir_name))[:length]
    b = np.load("{}/eval_results/eval.npy".format(dir_name))[:length]
    with open("{}/args.json".format(dir_name), 'r') as file:
        comment = json.load(file)["comment"]
    plt.figure(graph)
    plt.plot(a.mean(axis=1), label="empty reward {}".format(comment), color='r')
    plt.fill_between(range(len(a)), a.min(axis=1), a.max(axis=1),color='r', alpha=0.3)
    plt.plot(b.mean(axis=1), label="no empty reward {}".format(comment), color='g')
    plt.fill_between(range(len(b)), b.min(axis=1), b.max(axis=1),color='g', alpha=0.3)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend(loc=4)
    plt.grid(True)

# import pdb; pdb.set_trace()
# if args.option1:
#     plt.plot(a.mean(axis=1), label="empty reward {}".format(comment1), color='r')
#     plt.fill_between(range(len(a)), a.min(axis=1), a.max(axis=1),color='r', alpha=0.3)
#     plt.plot(b.mean(axis=1), label="no empty reward {}".format(comment1), color='g')
#     plt.fill_between(range(len(b)), b.min(axis=1), b.max(axis=1),color='g', alpha=0.3)
#
# if args.option2:
#     plt.plot(c.mean(axis=1), label="empty reward {}".format(comment2), color='r')
#     plt.fill_between(range(len(c)), c.min(axis=1), c.max(axis=1),color='r', alpha=0.3)
#     plt.plot(d.mean(axis=1), label="no empty reward {}".format(comment2), color='g')
#     plt.fill_between(range(len(d)), d.min(axis=1), d.max(axis=1),color='g', alpha=0.3)
#
# if args.option3:
#     plt.plot(e.mean(axis=1), label="empty reward {}".format(comment3), color='r')
#     plt.fill_between(range(len(e)), e.min(axis=1), e.max(axis=1),color='r', alpha=0.3)
#     plt.plot(f.mean(axis=1), label="no empty reward {}".format(comment3), color='g')
#     plt.fill_between(range(len(f)), f.min(axis=1), f.max(axis=1),color='g', alpha=0.3)
#
# if args.option4:
#     plt.plot(g.mean(axis=1), label="empty reward {}".format(comment4), color='r')
#     plt.fill_between(range(len(g)), g.min(axis=1), g.max(axis=1),color='r', alpha=0.3)
#     plt.plot(h.mean(axis=1), label="no empty reward {}".format(comment4), color='g')
#     plt.fill_between(range(len(h)), h.min(axis=1), h.max(axis=1),color='g', alpha=0.3)
#
# if args.option5:
#     plt.plot(i.mean(axis=1), label="empty reward added", color='r')
#     plt.fill_between(range(len(i)), i.min(axis=1), i.max(axis=1),color='r', alpha=0.3)
#     plt.plot(j.mean(axis=1), label="original reward", color='g')
#     plt.fill_between(range(len(j)), j.min(axis=1), j.max(axis=1),color='g', alpha=0.3)

# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.legend(loc=4)
# # plt.xticks([50,100,150,200,400,600,800,1000,1200,1600])
# plt.grid(True)
# # plt.show([l1,l2,l3,l4])
plt.show()

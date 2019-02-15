import json
import logging
import os
import shutil

from mecs import scheduler, config
from mecs.wholemap import WholeMap

logger = logging.getLogger(__name__)


def main():
    my_map = WholeMap(300, 300, 0.0003, 0.00001)
    server_capability = 30000  # clock per tick
    schedule_method = scheduler.RRScheduler().schedule
    my_map.add_server(150, 150, server_capability, schedule_method)
    log_dir = 'result'
    mobile_log = {}
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    os.mkdir(log_dir)

    for t in range(200001):
        # if t % 1000 == 0:
        #     logger.info(
        #         "================= << [%d,%d] second >> =================",
        #         t // 1000, t // 1000 + 1)
        #     logger.debug(json.dumps(mobile_log))
        #     mobile_log = {}
        mobile_log[t] = my_map.simulate_one_time(t)

    my_map.print_all_mobiles()


if __name__ == "__main__":
    config.initialize_mecs()
    main()

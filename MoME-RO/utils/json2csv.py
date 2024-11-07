import glob
from collections import OrderedDict
import json
import re


file_list = glob.glob('/home/gpuadmin/yujin/ro-llama/breast_target_volume_v2/Report/ex*.jsonl')
csv_list = []


for file_ in file_list:
    f = open(file_)
    data = json.load(f)
    f.close()

    log_name = file_.replace('.jsonl', '.csv')

    with open(log_name, "w") as log_file:
        log_file.write('id, plan\n')
        for idx, lines in enumerate(data):
            check_kr = lines['plan'].split('\n')
            remove_line = []
            for check_i in check_kr:
                check_kr_i = re.compile('[|가-힣]').findall(check_i)
                if len(check_kr_i) > 0:
                   remove_line.append(check_kr) 
            for r in remove_line:
                check_kr = check_kr.replace(r,'')
            plan = ' '.join(check_kr)
            # plan.replace(',', ';')
            data_ = '%s, %s'%(lines['id'], re.sub('[|⊙]+', '', plan))
            log_file.write('%s\n' % data_)  # save the message

# with open("/home/gpuadmin/yujin/ro-llama/library/ml-planner/data-bin/path_data/dummy.json", "w", encoding="utf-8") as f:
    
#     for idx, line_ in enumerate(dataset_train.medical_text):

#         # if idx > 100:
#         #     break

#         my_data = OrderedDict()
        
#         try:
            

#             line = line_.split("['")[-1].split("']")[0]
#             line = line.split("', '")
#             line = [l for l in line]

#             text = '; '.join(line)

#             if text == "[]":
#                 continue

#             my_data["text"] = text
#             my_data["timestamp"] = "2024-04"
#             my_data["url"] = "http://"
            
            
#             # f.write("%s\t%s\n"%(line[-3].replace('\n', ' '), line[-2].replace('\n', ' ')))

#         except:
#             pass

#         json.dump(my_data, f, ensure_ascii=False,) # ensure_ascii로 한글이 깨지지 않게 저장
#         f.write("\n") # json을 쓰는 것과 같지만, 여러 줄을 써주는 것이므로 "\n"을 붙여준다.

#         print(idx)
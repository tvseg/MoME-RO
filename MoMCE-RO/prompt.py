import pandas as pd
import re
from tqdm import tqdm
import openai

both_side = False
true_only = False
example_dir = '/home/gpuadmin/yujin/ro-llama/prostate_target_volume/20240131_prostate_prompt_초안.xlsx'


for s in [1]:

    # data
    flag_id, flag_pred = 'Unit No', 'None'
    report_example = pd.read_excel(example_dir)
    special = 'Summary note: {input}, Desired output: {output}. \n\n'
    col_special = ''
    for id in range(2):
        col_special += special.format(input=re.sub('[|가-힣]+', '', report_example.input[id]), output=report_example.gt_answer[id]) 
        
    col = ['summary']
    if s == 1:
        report_dir = '/home/gpuadmin/yujin/ro-llama/prostate_target_volume/Report/GN_add.xlsx'
        
    PROMPT_WITH_INPUT_FORMAT = '### Instruction:\n C Extract key information on prostate cancer treatment from the provided summary note. Summarize this information following the format of the desired output example. If information is missing, write N/A. Do not fabricate any details.\n\n### Example: {special}\n\n### Input: Summary note: {input}; Desired output: '


    # read excel
    report_all = pd.read_excel(report_dir)
    rep_id = report_all[flag_id]
    
    
    # load column
    for c in col:
        report_all.insert(report_all.columns.get_loc(c)+1, c + '_prompt', '')


    for iter, id in tqdm(enumerate(rep_id)):

        # if iter > 5:
        #     break

        i = report_all[report_all[flag_id]==id].index.values[0]

        translate_c = ''
        for c in col:
            
            input_kr = report_all._get_value(i, c)
            if input_kr != input_kr:
                output_en = ""
                break

            input_kr = re.sub('[|가-힣]+', '', input_kr)
            
            output_en_T = ''
            qustion = PROMPT_WITH_INPUT_FORMAT.format(special=col_special, input=input_kr) 

            # ChatGPT
            messages=[{"role": "user", "content": qustion}]
            output_en = ''
            while output_en == '':
                try:
                    output_en = openai.ChatCompletion.create(model="gpt-4-0613", messages=messages) #gpt-4-0314 gpt-3.5-turbo
                except:
                    print('server failed')
            output_en = output_en.choices[0].message["content"]

            output_en = output_en.split('Output: ')[-1]
            output_en = output_en.split('Output:')[-1]
            if output_en[0] == ' ':
                output_en = output_en[1:]
                
            if (len(output_en) > len(input_kr) * 10):
                print(len(output_en), output_en)
                print(len(input_kr), input_kr)
                output_en += ' ###### outlier'
            else:
                print('### output\n', output_en)
                
            if output_en.find('None') >= 0:
                output_en = ""

        report_all.update(pd.DataFrame({c + '_prompt': [output_en]}, index=[i]))

        report_all.to_excel(report_dir.replace('.xlsx', '_Prompted.xlsx')) #, index=False, encoding="utf-8-sig" #_en_SJP_instruction_


# scp -P 51014 /data/G/RO-GPT/breast_target_volume/231026_train_plan_vFormGPT4 slurm-user3@210.125.69.5:/scratch/slurm-user3/dataset/breast_target_volume/f_plan/
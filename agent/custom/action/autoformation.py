# function: 自动编队

# step 1 - 获取编队信息：查询当前激活的resource目录下的/pipeline/copilot文件夹内除了auto_formation.json和copilot_config.json之外的第3个json文件的文件名，并读取/resource/copilot-cache/文件夹内的同名json的内容，在其中找到"opers"对应的部分。
# 对opers进行分析，整理并记录 opers_num（opers包含的项数，可能为[1,5]的整数）及各oper对应的discs。后者需要将discs_selected中记录的index，通过查询/agent/operators.json中对应"name"的OPERATORS的"discs"[index]，取到对应的"ot_name"。

# step 2 - 根据整理后的opers信息依次完成选人（run task "自动编队-点击目标密探"，利用custom_reconition 通过ocr结果的similarly判断是否命中，比直接ocr更宽松；如果当前区域不存在，则下滑一屏继续识别）。
# 特别需要注意的是，队伍最多可选5人，需要先检查第一位是否为空（run recognition "自动编队-第一位为空"），根据结果后续处理不同。如第一位为空，则直接依次寻找并选择需要的opers即可。如第一位不为空，则在进行第一次选人时，需要考虑不同情况：当识别到目标密探时，需要额外检查指定区域（如在"roi" : [580,998,75,26]识别到目标密探，需额外检查[558,896,118,67]中是否存在文字"已上阵"（或为了提高识别率只expected:"上"），此处需要根据我给的例子计算出通用的roi_offset:[x_offset,y_offset,width_offset,height_offset]）。如存在已上阵，则说明当前一号位即为所需的一号位，不需要进行点击；否则需要在点击识别到的密探后，额外对"roi" : [69,425,42,51]进行依次点击，移除之前存在的一号位。后续可能存在的第2~5位队员都可做正常识别处理，不需考虑是否已上阵。

# step 3 - 完成选人后，需要依次对每个队员进行已生效命盘和discs ot_name的匹配度检测。
# 首先，需要从画面中提取出"生效中"命盘的文字信息。run recognition "自动编队-读取生效中命盘"，并对识别结果进行分析（有效结果个数可能为[0,3]的整数），获取有效结果的对应的roi。再根据有效结果的roi分别提取对应offset后的文字内容。如针对有效结果"roi" : [234,859,81,33]，需检测"roi" : [187,776,173,72]区域内的文字，请根据我提供的这个例子计算出通用的roi_offset
# 其次，检查opers discs的ot_name是否都存在于“生效中”（通过相似度进行比较）。如果有不存在的，则return fail。

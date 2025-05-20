# stego_encoders/arithmetic_stego.py
import torch
import torch.nn.functional as F
from utils.arithmetic import limit_past, bits2int, int2bits, num_same_from_beg

class ArithmeticStego:
    def __init__(self, model, top_k=50000, temperature=0.9, precision=16):
        self.model = model
        self.tokenizer = model.tokenizer
        self.device = model.device
        self.topk = top_k
        self.temp = temperature
        self.precision = precision
        self.max_val = 2 ** precision

    def embed(self, message, context):
        # 切片取最近的1022个token，转换成张量
        context = torch.tensor(context[-1022:], device=self.device, dtype=torch.long)
        # 算术编码区间上限（整数表示）
        max_val = 2**self.precision
        threshold = 2**(-self.precision)
        # 当前编码区间，左闭右开
        cur_interval = [0, max_val] # bottom inclusive, top exclusive
        # 用于迭代生成文本token
        prev = context
        output = context
        # 缓存Transformer的过去状态
        past = None
        
        with torch.no_grad():
            i = 0 # 已编码的消息比特数
            # 循环条件：没编码完所有消息比特
            while i < len(message):
                # 1. 预测token概率分布
                # 用模型根据当前输入token prev 和缓存 past 预测下一个token的logits
                outputs = self.model.model(prev.unsqueeze(0), past_key_values=past)
                logits = outputs.logits
                past = outputs.past_key_values
                past = limit_past(past)
                # 禁用一些特殊token出现（通过把对应logit置为极小值）
                logits[0, -1, -1] = -1e20 # endoftext token can't happen
                logits[0, -1, 628] = -1e20 # 2 newlines token can't happen
                # logits排序，得到按概率降序的token索引和概率
                logits, indices = logits[0, -1, :].sort(descending=True)
                logits = logits.double()
                # 根据温度调节概率分布
                logits_temp = logits / self.temp
                probs_temp = F.softmax(logits_temp, dim=0) # 概率向量
                log_probs = F.log_softmax(logits, dim=0)
                
                # 2. 编码消息比特段
                # Cutoff low probabilities that would be rounded to 0
                # 计算当前区间宽度 cur_int_range，用作概率范围映射
                cur_int_range = cur_interval[1]-cur_interval[0]
                cur_threshold = 1/cur_int_range # 阈值
                # 选取概率大于阈值的top-k词，截断低概率token以减少计算
                k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), self.topk)
                probs_temp_int = probs_temp[:k] # 保留top-k个概率
                
                # Rescale to correct range
                # 将概率重缩放成当前区间宽度的整数分布
                probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range

                # Round probabilities to integers given precision
                # 并累计概率成整数累积概率
                probs_temp_int = probs_temp_int.round().long()
                cum_probs = probs_temp_int.cumsum(0)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                # 如果累积概率大于当前区间宽度，去掉多余的元素
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]

                # Add any mass to the top if removing/rounding causes the total prob to be too small
                # 如果累积概率小于当前区间宽度，补齐到当前区间宽度
                cum_probs += cur_int_range-cum_probs[-1] # add

                # Get out resulting probabilities
                # 计算最终的概率分布
                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                # Convert to position in range
                # 将概率映射到当前区间
                cum_probs += cur_interval[0]

                # Get selected index based on binary fraction from message bits
                # 计算消息比特对应的整数索引 message_idx，找到对应的token索引 selection。
                message_bits = message[i:i+self.precision]
                if i+self.precision > len(message):
                    message_bits = message_bits + [0]*(i+self.precision-len(message))
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                # Calculate new range as ints
                # 计算新的区间范围
                new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                # 将当前区间转换成比特表示
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, self.precision)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, self.precision))) # -1 here because upper bound is exclusive

                # Consume most significant bits which are now fixed and update interval
                # 计算当前区间的比特表示，更新当前区间
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                i += num_bits_encoded

                new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
                new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

                cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
                cur_interval[1] = bits2int(reversed(new_int_top_bits))+1 # +1 here because upper bound is exclusive
                
                # Update history with new token
                # 更新历史生成token，加入输出序列。
                prev = indices[selection].view(1)
                output = torch.cat((output, prev))
                #print(enc.decode(prev.tolist()), message_bits[:num_bits_encoded])
                
                # For text->bits->text
                partial = self.tokenizer.decode(output[len(context):].tolist())
                # 生成token序列中检测是否遇到 <eos> 标记（结束符），则退出
                if '<eos>' in partial:
                    break

        return self.tokenizer.decode(output[len(context):].tolist())

    def decode(self, text, context):
        # input 和 context 是 token list
        input = self.tokenizer.encode(text)
        i = 0
        # 常见BPE修复：修复双换行被错误合并为 token 628 的情况
        while i < len(input):
            # 如果检测到 628，则拆分成两个 198 插回去
            if input[i] == 628:
                input[i] = 198
                input[i+1:i+1] = [198]
                i = i + 2
            else:
                i = i + 1 
        # 算术编码变量初始化
        # 限定 context 长度不超过 1022    
        context = torch.tensor(context[-1022:], device=self.device, dtype=torch.long)
        # 设置当前算术区间
        max_val = 2**self.precision
        threshold = 2**(-self.precision) # 最小的可表达概率
        cur_interval = [0, max_val] # 左闭右开   
        
        prev = context # 当前输入序列
        past = None # transformer 的 past key/value 缓存
        message = [] # 解码出的二进制消息
        with torch.no_grad():
            i = 0
            while i < len(input):
                # 使用 model 预测当前 token 的概率分布
                output = self.model.model(prev.unsqueeze(0), past_key_values=past)
                logits = output.logits
                past = output.past_key_values
                # limit_past 限制 past 长度（防止显存爆炸）
                past = limit_past(past)
                # 禁用一些特殊token出现（通过把对应logit置为极小值）
                logits[0, -1, -1] = -1e10 # endoftext can't happen
                logits[0, -1, 628] = -1e10 # 2 newlines can't happen
                # logits排序，得到按概率降序的token索引和概率
                logits, indices = logits[0, -1, :].sort(descending=True)
                logits = logits.double()
                # 根据温度调节概率分布
                logits_temp = logits / self.temp
                probs_temp = F.softmax(logits_temp, dim=0) # 概率向量
                
                # 2. 概率整数化 + 区间映射
                # 计算当前区间宽度 cur_int_range，用作概率范围映射
                # 当前区间宽度越小，所需 token 概率越大，因此删掉过小概率
                cur_int_range = cur_interval[1]-cur_interval[0]
                cur_threshold = 1/cur_int_range # 阈值
                # 选取概率大于阈值的top-k词，截断低概率token以减少计算
                k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), self.topk)
                probs_temp_int = probs_temp[:k] # 保留top-k个概率

                # 将概率重缩放成当前区间宽度的整数分布
                probs_temp_int = probs_temp_int/probs_temp_int.sum()*cur_int_range
                
                # 并累计概率成整数累积概率
                probs_temp_int = probs_temp_int.round().long()
                cum_probs = probs_temp_int.cumsum(0) # 累积概率
                
                # 如果累积概率(整数化后总和)大于当前区间宽度，去掉尾部多余元素
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]
                    k = overfill_index[0].item() # new
                # 如果累积概率小于当前区间宽度，补齐到当前区间宽度（将差值加到尾部）
                cum_probs += cur_int_range-cum_probs[-1] # add
                # 确定真实 token 在排序后的索引位置
                # 将概率映射到当前区间
                cum_probs += cur_interval[0]
                # rank:当前 token 在概率列表中的下标
                rank = (indices == input[i]).nonzero().item() # new
                # new
                # 如果这个 token 不在 top-k 中，需要尝试修复（BPE 错误）：
                if rank >= k:
                    true_token_text = self.tokenizer.decoder[input[i]]
                    for rank_idx in range(k):
                        prop_token_text = self.tokenizer.decoder[indices[rank_idx].item()]
                        # common case that is not caught
                        if input[i] == 128 and indices[rank_idx] == 198:
                            rank = rank_idx
                            input[i] = indices[rank_idx].item()
                            break
                        
                        # Is there a more likely prefix token that could be the actual token generated?
                        if len(prop_token_text) <= len(true_token_text) and \
                                prop_token_text == true_token_text[:len(prop_token_text)]:
                            rank = rank_idx
                            suffix = true_token_text[len(prop_token_text):]
                            suffix_tokens = self.tokenizer.encode(suffix) # a list
                            input[i] = indices[rank_idx].item()
                            input[i+1:i+1] = suffix_tokens # insert suffix tokens into list
                            break

                        # Is there a more likely longer token that could be the actual token generated?
                        elif len(prop_token_text) > len(true_token_text) and \
                                true_token_text == prop_token_text[:len(true_token_text)]:
                            whole_text = true_token_text
                            num_extra = 1
                            while len(whole_text) < len(prop_token_text):
                                whole_text += self.tokenizer.decoder[input[i+num_extra]]
                                num_extra += 1
                            if prop_token_text == whole_text[:len(prop_token_text)]:
                                rank = rank_idx
                                input[i] = indices[rank_idx].item()
                                for j in range(1, num_extra):
                                    del input[i+j]

                                if len(whole_text) > len(prop_token_text):
                                    suffix = whole_text[len(prop_token_text):]
                                    suffix_tokens = self.tokenizer.encode(suffix) # a list
                                    input[i+1:i+1] = suffix_tokens # insert suffix tokens into list
                                break
                    else:
                        print('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, input[i], text))
                        rank = 0
                
                # 区间更新与比特输出
                selection = rank
                # 当前 token 对应的概率区间[new_int_bottom, new_int_top)
                new_int_bottom = cum_probs[selection-1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]
                # 将当前区间转换成比特表示
                # 找出新上下限的公共前缀比特数，即是可确定的比特
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, self.precision)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top-1, self.precision))) # -1 here because 上界开区间
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                # 将这部分固定比特加入 message，最后一个 token 直接加全部位
                if i == len(input)-1:
                    new_bits = new_int_bottom_bits_inc
                else:
                    new_bits = new_int_top_bits_inc[:num_bits_encoded]
                message += new_bits
                # 区间压缩、prev 更新
                # 去掉固定比特后更新区间，准备下一轮
                new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0]*num_bits_encoded
                new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1]*num_bits_encoded

                cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
                cur_interval[1] = bits2int(reversed(new_int_top_bits))+1 # +1 here because upper bound is exclusive
                
                # 将当前 token 作为下一个 prev
                prev = torch.tensor([input[i]], device=self.device, dtype=torch.long)
                #print(enc.decode([inp[i]]), new_bits)
                i += 1
        return message        
    
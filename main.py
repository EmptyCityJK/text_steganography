# main.py
import argparse
import bitarray
from utils.arithmetic import encode_context
from models.bert_wrapper import BERTWrapper
from stego_encoders.edit_stego import EditStego
from models.gpt2_wrapper import GPT2Wrapper
from stego_encoders.bins_stego import BinsStego
from stego_encoders.huffman_stego import HuffmanStego
from stego_encoders.perfect_tree_stego import PerfectTreeStego
from stego_encoders.arithmetic_stego import ArithmeticStego

def run_edit(args):
    model = BERTWrapper(model_name=args.model)
    stego = EditStego(model, k=args.k)

    cover_text = stego.embed(secret_text=args.secret_text, context=args.context)
    print("\n[Generated Cover Text]:\n", cover_text)
    secret = stego.decode(context=args.context, cover_text=cover_text)
    print("\n[Recovered Secret Message]:\n", secret)
        

def run_bins(args):
    model = GPT2Wrapper(model_name=args.model)
    stego = BinsStego(model, bit_width=(args.k - 1).bit_length())

    result_text = stego.embed(args.secret_text, args.context)
    print("\n[Generated Cover Text]:\n", result_text)
    secret = stego.decode(context=args.context, cover_text=result_text)
    print("\n[Recovered Secret Message]:\n", secret)
        
def run_huffman(args):
    model = GPT2Wrapper(model_name=args.model)
    stego = HuffmanStego(model, top_k=args.k)

    result_text = stego.embed(args.secret_text, args.context)
    print("\n[Generated Cover Text]:\n", result_text)
    secret = stego.decode(context=args.context, cover_text=result_text)
    print("\n[Recovered Secret Message]:\n", secret)
        
def run_huffman_fixed(args):
    model = GPT2Wrapper(model_name=args.model)
    stego = PerfectTreeStego(model, bit_width=(args.k - 1).bit_length())

    result_text = stego.embed(args.secret_text, args.context)
    print("\n[Generated Cover Text]:\n", result_text)
    secret = stego.decode(args.context, result_text)
    print("\n[Recovered Secret Message]:\n", secret)

def run_arithmetic(args):
    model_wrapper = GPT2Wrapper(model_name=args.model)
    enc = model_wrapper.tokenizer  # 获取tokenizer
    model = model_wrapper.model  # 获取模型
    # context_tokens 是 list[int]，带有 <|endoftext|> 结尾token
    context_tokens = encode_context(args.context, enc)
    ba = bitarray.bitarray()
    message_str = args.secret_text
    ba.frombytes(message_str.encode('utf-8'))
    # message 为 Python列表（[0,1,1,0,1,0,...]）
    message = ba.tolist()
    
    stego = ArithmeticStego(model_wrapper, top_k=args.k, temperature=args.temperature, precision=args.precision)

    result_text = stego.embed(message=message, context=context_tokens)
    print("\n[Generated Cover Text]:\n", result_text)
    secret_bit = stego.decode(text=result_text, context=context_tokens)
    # print("\n[Recovered Secret Bits]:\n", secret_bit)
    # 最后将密文比特映射回原始密文
    secret_bit = [bool(item) for item in secret_bit]
    ba = bitarray.bitarray(secret_bit)
    secret = ba.tobytes().decode('utf-8', 'ignore')
    print("\n[Recovered Secret Message]:\n", secret)



def main():
    parser = argparse.ArgumentParser(description="Text Steganography System")

    parser.add_argument("--task", type=str, choices=["edit", "bins", "huffman", "huffman_fixed", "arithmetic"], required=True, help="Task type")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--k", type=int, default=4, help="Top-k candidate tokens used per mask")
    parser.add_argument("--temperature", type=float, default=0.8, help="Softmax temperature (lower = sharper distribution)")
    parser.add_argument("--precision", type=int, default=16, help="Precision for arithmetic steganography 26~40")
    
    parser.add_argument("--context", type=str, default="Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission.", help="LM context (shared)")
    parser.add_argument("--secret_text", type=str, default="This is a very secret message!", help="Secret message (for encrypt)")

    args = parser.parse_args()

    if args.task == "edit":
        run_edit(args)
    elif args.task == "bins":
        run_bins(args)
    elif args.task == "huffman":
        run_huffman(args)
    elif args.task == "huffman_fixed":
        run_huffman_fixed(args)
    elif args.task == "arithmetic":
        run_arithmetic(args) 
    else:
        raise NotImplementedError(f"Task '{args.task}' not supported yet.")



if __name__ == "__main__":
    main()

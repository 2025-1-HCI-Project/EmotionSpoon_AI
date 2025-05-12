import torch, os
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline

from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class SentimentAnalyzer:
    def __init__(self):
        # .env 파일 로드
        load_dotenv()

        # .env 파일에 있는 hugging face token 로드
        token = os.getenv('HUGGING_FACE_HUB_TOKEN')

        GPU_NUM = 0
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        print('Device for LLM:', device)

        if torch.cuda.is_available():
            torch.cuda.set_device(device)

            print('Current cuda device:', torch.cuda.current_device())
            print('Count of using GPUs:', torch.cuda.device_count())

        model_id = "meta-llama/Llama-3.2-1B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

        text_generation_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=1e-6,
            return_full_text=False,
            max_new_tokens=64
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        PROMPT = \
        '''
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a psychotherapist to analyze sentiment in Text.

Guideline:
    - Extract one of main emotion from Text.
    - The emotion cannot be general sentiment such as good, bad, positive, and negative.

Remember, the emotion must be detailed as you can.

Print the main emotion without your rationale.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Text: {input}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
Sentiment:
        '''.strip()

        prompt_template = PromptTemplate(
            input_variables=["input"],
            template=PROMPT,
        )

        self.llm_chain = prompt_template | llm | StrOutputParser()

        print("LLM chain loaded")
    
    def analyze(self, text):
        answer = self.llm_chain.invoke({"input": text})
        return answer.split("\n")[0].strip()
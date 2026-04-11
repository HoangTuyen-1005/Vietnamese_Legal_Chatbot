import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class LocalLLMGenerator:
    def __init__(self, model_id="microsoft/Phi-4-mini-instruct"):
        print("Loading Model...")

        # Quantization config để chạy gọn VRAM hơn
        quant_config = None
        model_kwargs = {}

        if device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs = {
                "device_map": "auto",
                "quantization_config": quant_config,
                "torch_dtype": torch.float16,
            }
        else:
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float32,
            }

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id
        )

        # Phi đôi khi cần pad_token fallback
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=None,
            top_p=None,
            repetition_penalty=1.1,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        print("Model Loaded.")

    def generate_answer(self, query, full_contexts):
        combined_context = "\n\n---\n\n".join(full_contexts)

        messages = [
            {
                "role": "system",
                "content": """
Bạn là AI hỗ trợ phân tích pháp luật tiếng Việt.

NHIỆM VỤ:
- Chỉ được trả lời dựa trên phần [VĂN BẢN LUẬT] được cung cấp.
- Không được dùng kiến thức ngoài văn bản.
- Không được bịa điều luật, khoản, mức phạt, hay suy diễn vượt quá nội dung văn bản.
- Phải kiểm tra xem [VĂN BẢN LUẬT] có thực sự liên quan trực tiếp đến [CÂU HỎI] hay không.

QUY TẮC BẮT BUỘC:
1. Trước khi trả lời, phải tự kiểm tra:
- Văn bản được cung cấp có nói đúng chủ đề mà câu hỏi đang hỏi không?
- Có đúng hành vi pháp lý đang được hỏi không?
- Có đúng loại trách nhiệm pháp lý đang được hỏi không?

2. Nếu văn bản không liên quan trực tiếp, hoặc không đủ căn cứ để trả lời câu hỏi, phải dừng lại và trả lời đúng nguyên văn:
"Xin lỗi, dữ liệu pháp luật hiện tại của tôi không đề cập đến vấn đề này."

3. Không được cố gắng trích một đoạn chỉ vì nó có vài từ giống câu hỏi.

4. Không được kết luận từ điều luật không cùng bản chất hành vi.
Ví dụ: hỏi về lừa đảo chiếm đoạt thì không được dùng điều luật về rửa tiền, trộm cắp, hay tội khác nếu văn bản không khớp.

5. Nếu tìm thấy nội dung phù hợp, phải trả lời đúng định dạng sau:

Trích dẫn nguyên văn:
- [trích nguyên văn đúng đoạn liên quan nhất, giữ nguyên câu chữ]

Căn cứ pháp lý:
- [ghi rõ Điều, Khoản, Điểm nếu có trong chính văn bản được cung cấp]

Kết luận:
- [chỉ kết luận trong phạm vi phần trích dẫn]
- [nếu văn bản chưa đủ để xác định dứt khoát, phải viết rõ: "Chưa đủ căn cứ từ văn bản được cung cấp để kết luận cụ thể hơn."]

6. Không được nói về mức án cụ thể nếu phần văn bản trích ra không nêu rõ mức án đó.

7. Không được đảo vai người hỏi, nạn nhân, người vi phạm nếu văn bản không nêu rõ.

ƯU TIÊN:
- Độ chính xác pháp lý quan trọng hơn việc trả lời dài.
- Nếu thiếu căn cứ, phải từ chối thay vì suy đoán.
"""
            },
            {
                "role": "user",
                "content": f"""
[VĂN BẢN LUẬT]:
{combined_context}

[CÂU HỎI]:
{query}

Yêu cầu:
- Chỉ dùng đúng [VĂN BẢN LUẬT] ở trên.
- Nếu không có đoạn nào liên quan trực tiếp và đủ căn cứ, hãy trả lời đúng nguyên văn:
"Xin lỗi, dữ liệu pháp luật hiện tại của tôi không đề cập đến vấn đề này."
"""
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.pipe(prompt)
        response = outputs[0]["generated_text"][len(prompt):].strip()

        return response
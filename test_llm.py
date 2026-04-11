from generator import LocalLLMGenerator
import time

def test_member_3():
    generator = LocalLLMGenerator()
    
    print("\n" + "="*50)
    print("🧪 BÀI TEST SỐ 1: KIỂM TRA ĐỘ CHÍNH XÁC")
    query_1 = "Tội cướp tài sản bị phạt tù bao nhiêu năm?"
    context_1 = [
        "Điều 168. Tội cướp tài sản\n1. Người nào dùng vũ lực, đe dọa dùng vũ lực ngay tức khắc... thì bị phạt tù từ 03 năm đến 10 năm."
    ]
    
    start_time = time.time()
    ans_1 = generator.generate_answer(query_1, context_1)
    print(f"\n[Trả lời trong {time.time() - start_time:.2f}s]:\n{ans_1}")
    
    print("\n" + "="*50)
    print("🧪 BÀI TEST SỐ 2: KIỂM TRA CHỐNG ẢO GIÁC (HALLUCINATION)")
    query_2 = "Đi xe máy vượt đèn đỏ bị phạt bao nhiêu tiền?"
    # Đưa vào một context hoàn toàn không liên quan (vd: luật Cướp tài sản)
    context_2 = context_1 
    
    ans_2 = generator.generate_answer(query_2, context_2)
    print(f"\n[Trả lời chống ảo giác]:\n{ans_2}")
    print("="*50)

if __name__ == "__main__":
    test_member_3()
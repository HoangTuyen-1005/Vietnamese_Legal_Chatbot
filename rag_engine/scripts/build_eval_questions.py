from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rag_engine.retrieval.refusal_policy import REFUSAL_ANSWER


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data_pipeline" / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data_pipeline" / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "rag_engine" / "eval" / "eval_questions.json"
SOURCE_FIELDS = ("so_hieu", "dieu", "khoan", "diem")

LAW_NAMES = {
    "91/2015/QH13": "Bộ luật Dân sự 2015",
    "01/VBHN-VPQH": "Bộ luật Hình sự 2015, sửa đổi bổ sung 2017",
    "31/2024/QH15": "Luật Đất đai 2024",
    "32/2024/QH15": "Luật Các tổ chức tín dụng 2024",
    "148/2025/QH15": "Luật Chuyển đổi số 2025",
}


def src(
    so_hieu: str,
    dieu: str,
    khoan: str | None = None,
    diem: str | None = None,
) -> dict[str, str | None]:
    return {
        "so_hieu": so_hieu,
        "dieu": dieu,
        "khoan": khoan,
        "diem": diem,
    }


def item(
    question: str,
    category: str,
    difficulty: str,
    query_type: str,
    source: dict[str, str | None] | None,
    note: str,
) -> dict[str, Any]:
    return {
        "question": question,
        "category": category,
        "difficulty": difficulty,
        "query_type": query_type,
        "source": source,
        "notes": note,
    }


CURATED_ITEMS = [
    item("Các nguyên tắc cơ bản của pháp luật dân sự gồm những gì?", "civil_law_principle", "easy", "principle_lookup", src("91/2015/QH13", "Điều 3"), "Nguyên tắc nền tảng của BLDS."),
    item("Khi không có thỏa thuận và pháp luật không quy định thì có được áp dụng tập quán không?", "civil_law_general", "medium", "article_lookup", src("91/2015/QH13", "Điều 5", "Khoản 2"), "Áp dụng tập quán trong quan hệ dân sự."),
    item("Quyền dân sự được xác lập từ những căn cứ nào?", "civil_law_general", "easy", "cases_circumstances", src("91/2015/QH13", "Điều 8"), "Căn cứ xác lập quyền dân sự."),
    item("Khi quyền dân sự bị xâm phạm, chủ thể có thể yêu cầu những phương thức bảo vệ nào?", "civil_law_general", "medium", "cases_circumstances", src("91/2015/QH13", "Điều 11"), "Phương thức bảo vệ quyền dân sự."),
    item("Năng lực pháp luật dân sự của cá nhân là gì?", "civil_law_capacity", "easy", "definition_lookup", src("91/2015/QH13", "Điều 16", "Khoản 1"), "Định nghĩa năng lực pháp luật dân sự."),
    item("Năng lực hành vi dân sự của cá nhân là gì?", "civil_law_capacity", "easy", "definition_lookup", src("91/2015/QH13", "Điều 19"), "Định nghĩa năng lực hành vi dân sự."),
    item("Thế nào là người thành niên?", "civil_law_capacity", "easy", "definition_lookup", src("91/2015/QH13", "Điều 20", "Khoản 1"), "Định nghĩa người thành niên."),
    item("Người chưa đủ sáu tuổi có tự mình xác lập giao dịch dân sự không?", "civil_law_capacity", "medium", "article_lookup", src("91/2015/QH13", "Điều 21", "Khoản 2"), "Giao dịch của người chưa đủ sáu tuổi."),
    item("Trường hợp nào Tòa án tuyên bố một người mất năng lực hành vi dân sự?", "civil_law_capacity", "medium", "cases_circumstances", src("91/2015/QH13", "Điều 22", "Khoản 1"), "Tuyên bố mất năng lực hành vi dân sự."),
    item("Việc sử dụng hình ảnh của cá nhân phải đáp ứng điều kiện gì?", "civil_law_personal_rights", "medium", "conditions", src("91/2015/QH13", "Điều 32", "Khoản 1"), "Quyền của cá nhân đối với hình ảnh."),
    item("Cá nhân có quyền được bảo đảm an toàn về tính mạng, sức khỏe, thân thể như thế nào?", "civil_law_personal_rights", "easy", "article_lookup", src("91/2015/QH13", "Điều 33"), "Quyền sống và an toàn thân thể."),
    item("Cá nhân được xác định lại giới tính trong trường hợp nào?", "civil_law_personal_rights", "medium", "cases_circumstances", src("91/2015/QH13", "Điều 36"), "Quyền xác định lại giới tính."),
    item("Thư tín, điện thoại, dữ liệu điện tử của cá nhân có được bảo đảm bí mật không?", "civil_law_personal_rights", "medium", "article_lookup", src("91/2015/QH13", "Điều 38", "Khoản 3"), "Bí mật đời sống riêng tư."),
    item("Nơi cư trú của cá nhân được xác định theo tiêu chí nào?", "civil_law_personal_rights", "easy", "definition_lookup", src("91/2015/QH13", "Điều 40", "Khoản 1"), "Nơi cư trú của cá nhân."),
    item("Cá nhân làm người giám hộ phải có những điều kiện gì?", "civil_law_guardianship", "medium", "conditions", src("91/2015/QH13", "Điều 49"), "Điều kiện làm người giám hộ."),
    item("Ai có quyền cử người giám sát việc giám hộ?", "civil_law_guardianship", "hard", "procedure_lookup", src("91/2015/QH13", "Điều 51"), "Giám sát việc giám hộ."),
    item("Khi nào một người có thể bị Tòa án tuyên bố là đã chết?", "civil_law_capacity", "hard", "cases_circumstances", src("91/2015/QH13", "Điều 71", "Khoản 1"), "Các trường hợp tuyên bố chết."),
    item("Tài sản theo Bộ luật Dân sự bao gồm những gì?", "civil_law_property", "easy", "definition_lookup", src("91/2015/QH13", "Điều 105", "Khoản 1"), "Khái niệm tài sản."),
    item("Giao dịch dân sự là gì?", "civil_law_transaction", "easy", "definition_lookup", src("91/2015/QH13", "Điều 116"), "Định nghĩa giao dịch dân sự."),
    item("Giao dịch dân sự có hiệu lực khi đáp ứng điều kiện nào?", "civil_law_transaction", "medium", "conditions", src("91/2015/QH13", "Điều 117", "Khoản 1"), "Điều kiện có hiệu lực của giao dịch dân sự."),

    item("Tội phạm là gì theo Bộ luật Hình sự?", "criminal_law_general", "easy", "definition_lookup", src("01/VBHN-VPQH", "Điều 8", "Khoản 1"), "Khái niệm tội phạm."),
    item("Tội phạm được phân loại như thế nào?", "criminal_law_general", "medium", "cases_circumstances", src("01/VBHN-VPQH", "Điều 9"), "Phân loại tội phạm."),
    item("Người từ đủ bao nhiêu tuổi phải chịu trách nhiệm hình sự về mọi tội phạm?", "criminal_law_responsibility", "easy", "article_lookup", src("01/VBHN-VPQH", "Điều 12", "Khoản 1"), "Tuổi chịu trách nhiệm hình sự."),
    item("Người phạm tội do dùng rượu, bia có phải chịu trách nhiệm hình sự không?", "criminal_law_responsibility", "medium", "article_lookup", src("01/VBHN-VPQH", "Điều 13"), "Trách nhiệm khi dùng rượu, bia hoặc chất kích thích."),
    item("Chuẩn bị phạm tội là gì?", "criminal_law_general", "easy", "definition_lookup", src("01/VBHN-VPQH", "Điều 14", "Khoản 1"), "Khái niệm chuẩn bị phạm tội."),
    item("Phạm tội chưa đạt là gì?", "criminal_law_general", "easy", "definition_lookup", src("01/VBHN-VPQH", "Điều 15"), "Khái niệm phạm tội chưa đạt."),
    item("Đồng phạm là gì?", "criminal_law_general", "easy", "definition_lookup", src("01/VBHN-VPQH", "Điều 17", "Khoản 1"), "Khái niệm đồng phạm."),
    item("Che giấu tội phạm được quy định ở đâu?", "criminal_law_article", "easy", "article_lookup", src("01/VBHN-VPQH", "Điều 18"), "Định vị điều luật."),
    item("Phòng vệ chính đáng là gì?", "criminal_law_general", "medium", "definition_lookup", src("01/VBHN-VPQH", "Điều 22", "Khoản 1"), "Phòng vệ chính đáng."),
    item("Tình thế cấp thiết là gì?", "criminal_law_general", "medium", "definition_lookup", src("01/VBHN-VPQH", "Điều 23", "Khoản 1"), "Tình thế cấp thiết."),
    item("Thời hiệu truy cứu trách nhiệm hình sự được quy định thế nào?", "criminal_law_responsibility", "medium", "article_lookup", src("01/VBHN-VPQH", "Điều 27"), "Thời hiệu truy cứu trách nhiệm hình sự."),
    item("Những căn cứ nào có thể miễn trách nhiệm hình sự?", "criminal_law_responsibility", "hard", "cases_circumstances", src("01/VBHN-VPQH", "Điều 29"), "Căn cứ miễn trách nhiệm hình sự."),
    item("Hình phạt là gì?", "criminal_law_penalty", "easy", "definition_lookup", src("01/VBHN-VPQH", "Điều 30"), "Khái niệm hình phạt."),
    item("Các hình phạt chính đối với người phạm tội gồm những hình phạt nào?", "criminal_law_penalty", "medium", "cases_circumstances", src("01/VBHN-VPQH", "Điều 32", "Khoản 1"), "Các hình phạt đối với người phạm tội."),
    item("Phạt tiền được áp dụng như hình phạt chính với đối tượng nào?", "criminal_law_penalty", "medium", "penalty_lookup", src("01/VBHN-VPQH", "Điều 35", "Khoản 1"), "Quy định về phạt tiền."),
    item("Tù có thời hạn tối thiểu và tối đa là bao lâu?", "criminal_law_penalty", "easy", "penalty_lookup", src("01/VBHN-VPQH", "Điều 38", "Khoản 1"), "Mức tù có thời hạn."),
    item("Tử hình không áp dụng đối với những người nào?", "criminal_law_penalty", "medium", "cases_circumstances", src("01/VBHN-VPQH", "Điều 40", "Khoản 2"), "Trường hợp không áp dụng tử hình."),
    item("Các tình tiết giảm nhẹ trách nhiệm hình sự gồm những gì?", "criminal_law_penalty", "hard", "cases_circumstances", src("01/VBHN-VPQH", "Điều 51", "Khoản 1"), "Tình tiết giảm nhẹ."),
    item("Tái phạm hoặc phạm tội có tổ chức có phải là tình tiết tăng nặng không?", "criminal_law_penalty", "medium", "article_lookup", src("01/VBHN-VPQH", "Điều 52", "Khoản 1"), "Tình tiết tăng nặng."),
    item("Điều kiện để người bị phạt tù được hưởng án treo là gì?", "criminal_law_penalty", "hard", "conditions", src("01/VBHN-VPQH", "Điều 65", "Khoản 1"), "Điều kiện hưởng án treo."),

    item("Người sử dụng đất gồm những đối tượng nào?", "land_law_general", "easy", "definition_lookup", src("31/2024/QH15", "Điều 4"), "Người sử dụng đất."),
    item("Nguyên tắc sử dụng đất theo Luật Đất đai 2024 là gì?", "land_law_principle", "easy", "principle_lookup", src("31/2024/QH15", "Điều 5"), "Nguyên tắc sử dụng đất."),
    item("Đất đai được phân loại thành những nhóm nào?", "land_law_general", "easy", "definition_lookup", src("31/2024/QH15", "Điều 9"), "Phân loại đất."),
    item("Những hành vi nào bị nghiêm cấm trong lĩnh vực đất đai?", "land_law_prohibited_acts", "easy", "prohibited_acts", src("31/2024/QH15", "Điều 11"), "Hành vi bị nghiêm cấm."),
    item("Đất đai thuộc sở hữu của ai?", "land_law_principle", "easy", "article_lookup", src("31/2024/QH15", "Điều 12"), "Sở hữu đất đai."),
    item("Công dân có quyền gì đối với đất đai?", "land_law_rights", "medium", "article_lookup", src("31/2024/QH15", "Điều 23"), "Quyền của công dân đối với đất đai."),
    item("Người sử dụng đất có quyền chung nào?", "land_law_rights", "medium", "article_lookup", src("31/2024/QH15", "Điều 26"), "Quyền chung của người sử dụng đất."),
    item("Người sử dụng đất có nghĩa vụ chung gì?", "land_law_rights", "medium", "article_lookup", src("31/2024/QH15", "Điều 31"), "Nghĩa vụ chung của người sử dụng đất."),
    item("Điều kiện thực hiện quyền chuyển nhượng, tặng cho, thế chấp quyền sử dụng đất là gì?", "land_law_rights", "hard", "conditions", src("31/2024/QH15", "Điều 45"), "Điều kiện thực hiện quyền của người sử dụng đất."),
    item("Hộ gia đình, cá nhân được chuyển đổi quyền sử dụng đất nông nghiệp khi nào?", "land_law_rights", "medium", "conditions", src("31/2024/QH15", "Điều 47"), "Điều kiện chuyển đổi quyền sử dụng đất nông nghiệp."),
    item("Nguyên tắc lập quy hoạch, kế hoạch sử dụng đất là gì?", "land_law_planning", "medium", "principle_lookup", src("31/2024/QH15", "Điều 60"), "Nguyên tắc lập quy hoạch."),
    item("Những trường hợp nào Nhà nước thu hồi đất để phát triển kinh tế - xã hội vì lợi ích quốc gia, công cộng?", "land_law_recovery", "hard", "cases_circumstances", src("31/2024/QH15", "Điều 79"), "Thu hồi đất để phát triển kinh tế - xã hội."),
    item("Căn cứ và điều kiện thu hồi đất vì mục đích quốc phòng, an ninh hoặc phát triển kinh tế - xã hội là gì?", "land_law_recovery", "hard", "conditions", src("31/2024/QH15", "Điều 80"), "Căn cứ, điều kiện thu hồi đất."),
    item("Các trường hợp thu hồi đất do vi phạm pháp luật về đất đai gồm những gì?", "land_law_recovery", "medium", "cases_circumstances", src("31/2024/QH15", "Điều 81"), "Thu hồi đất do vi phạm pháp luật."),
    item("Trình tự, thủ tục bồi thường, hỗ trợ, tái định cư khi thu hồi đất được quy định ở điều nào?", "land_law_recovery", "medium", "procedure_lookup", src("31/2024/QH15", "Điều 87"), "Thủ tục bồi thường, hỗ trợ, tái định cư."),
    item("Nguyên tắc bồi thường, hỗ trợ, tái định cư khi Nhà nước thu hồi đất là gì?", "land_law_recovery", "medium", "principle_lookup", src("31/2024/QH15", "Điều 91"), "Nguyên tắc bồi thường, hỗ trợ, tái định cư."),
    item("Điều kiện được bồi thường về đất khi Nhà nước thu hồi đất vì mục đích quốc phòng, an ninh là gì?", "land_law_recovery", "hard", "conditions", src("31/2024/QH15", "Điều 95"), "Điều kiện được bồi thường về đất."),
    item("Định giá đất phải tuân theo nguyên tắc nào?", "land_law_principle", "medium", "principle_lookup", src("31/2024/QH15", "Điều 158"), "Nguyên tắc định giá đất."),
    item("Bảng giá đất được xây dựng và áp dụng như thế nào?", "land_law_principle", "medium", "procedure_lookup", src("31/2024/QH15", "Điều 159"), "Bảng giá đất."),
    item("Tách thửa đất, hợp thửa đất phải bảo đảm những yêu cầu gì?", "land_law_certificate", "hard", "conditions", src("31/2024/QH15", "Điều 220"), "Tách thửa, hợp thửa đất."),
    item("Tranh chấp đất đai có bắt buộc hòa giải tại UBND cấp xã không?", "land_law_dispute", "medium", "procedure_lookup", src("31/2024/QH15", "Điều 235"), "Hòa giải tranh chấp đất đai."),
    item("Người vi phạm pháp luật về đất đai bị xử lý như thế nào?", "land_law_penalty", "medium", "penalty_lookup", src("31/2024/QH15", "Điều 239"), "Xử lý vi phạm pháp luật về đất đai."),

    item("Tổ chức tín dụng là gì?", "banking_law_definition", "easy", "definition_lookup", src("32/2024/QH15", "Điều 4", "Khoản 38"), "Giải thích từ ngữ trong luật các tổ chức tín dụng."),
    item("Tổ chức có được sử dụng cụm từ ngân hàng khi không phải tổ chức tín dụng không?", "banking_law_operation", "medium", "article_lookup", src("32/2024/QH15", "Điều 5"), "Sử dụng từ ngữ liên quan đến hoạt động ngân hàng."),
    item("Tổ chức tín dụng được hoạt động ngân hàng trong phạm vi nào?", "banking_law_operation", "easy", "article_lookup", src("32/2024/QH15", "Điều 8"), "Quyền hoạt động ngân hàng."),
    item("Tổ chức tín dụng phải bảo mật thông tin khách hàng như thế nào?", "banking_law_operation", "medium", "article_lookup", src("32/2024/QH15", "Điều 13"), "Bảo mật thông tin."),
    item("An toàn dữ liệu và hoạt động liên tục của tổ chức tín dụng được yêu cầu ra sao?", "banking_law_operation", "medium", "conditions", src("32/2024/QH15", "Điều 14"), "An toàn dữ liệu và hoạt động liên tục."),
    item("Những hành vi nào bị nghiêm cấm trong hoạt động tổ chức tín dụng?", "banking_law_prohibited_acts", "easy", "prohibited_acts", src("32/2024/QH15", "Điều 15"), "Hành vi bị nghiêm cấm."),
    item("Cơ quan nào có thẩm quyền cấp, sửa đổi, thu hồi Giấy phép của tổ chức tín dụng?", "banking_law_license", "medium", "procedure_lookup", src("32/2024/QH15", "Điều 27"), "Thẩm quyền cấp, sửa đổi, thu hồi giấy phép."),
    item("Điều kiện cấp Giấy phép đối với tổ chức tín dụng là gì?", "banking_law_license", "hard", "conditions", src("32/2024/QH15", "Điều 29"), "Điều kiện cấp giấy phép."),
    item("Hồ sơ, thủ tục cấp Giấy phép tổ chức tín dụng được quy định ở đâu?", "banking_law_license", "medium", "procedure_lookup", src("32/2024/QH15", "Điều 30"), "Hồ sơ, thủ tục cấp giấy phép."),
    item("Thời hạn cấp Giấy phép cho tổ chức tín dụng là bao lâu?", "banking_law_license", "medium", "article_lookup", src("32/2024/QH15", "Điều 31"), "Thời hạn cấp giấy phép."),
    item("Những trường hợp nào không được đảm nhiệm chức vụ trong tổ chức tín dụng?", "banking_law_governance", "hard", "cases_circumstances", src("32/2024/QH15", "Điều 42"), "Trường hợp không được đảm nhiệm chức vụ."),
    item("Tỷ lệ sở hữu cổ phần của cổ đông trong tổ chức tín dụng được giới hạn như thế nào?", "banking_law_governance", "medium", "article_lookup", src("32/2024/QH15", "Điều 63"), "Tỷ lệ sở hữu cổ phần."),
    item("Nội dung hoạt động được phép của tổ chức tín dụng gồm những gì?", "banking_law_operation", "medium", "article_lookup", src("32/2024/QH15", "Điều 99"), "Nội dung hoạt động được phép."),
    item("Tổ chức tín dụng công khai báo cáo tài chính trong thời hạn nào?", "banking_law_operation", "medium", "article_lookup", src("32/2024/QH15", "Điều 154"), "Công khai báo cáo tài chính."),
    item("Khi nào tổ chức tín dụng được thực hiện can thiệp sớm?", "banking_law_operation", "hard", "cases_circumstances", src("32/2024/QH15", "Điều 156"), "Thực hiện can thiệp sớm."),

    item("Luật Chuyển đổi số áp dụng với đối tượng nào?", "digital_law_general", "easy", "article_lookup", src("148/2025/QH15", "Điều 2"), "Đối tượng áp dụng."),
    item("Chuyển đổi số là gì?", "digital_law_definition", "easy", "definition_lookup", src("148/2025/QH15", "Điều 3", "Khoản 1"), "Giải thích từ ngữ."),
    item("Hoạt động chuyển đổi số gồm những hoạt động nào?", "digital_law_general", "medium", "cases_circumstances", src("148/2025/QH15", "Điều 4"), "Hoạt động chuyển đổi số."),
    item("Những hành vi nào bị nghiêm cấm trong chuyển đổi số?", "digital_law_prohibited_acts", "easy", "prohibited_acts", src("148/2025/QH15", "Điều 5"), "Hành vi bị nghiêm cấm."),
    item("Nguyên tắc chuyển đổi số là gì?", "digital_law_principle", "easy", "principle_lookup", src("148/2025/QH15", "Điều 6"), "Nguyên tắc chuyển đổi số."),
    item("Hệ thống số phải đáp ứng yêu cầu tối thiểu nào?", "digital_law_conditions", "medium", "conditions", src("148/2025/QH15", "Điều 8"), "Yêu cầu tối thiểu đối với hệ thống số."),
    item("Thử nghiệm có kiểm soát trong chuyển đổi số được quy định như thế nào?", "digital_law_procedure", "medium", "procedure_lookup", src("148/2025/QH15", "Điều 28"), "Thử nghiệm có kiểm soát."),
    item("Quyền con người, quyền công dân trên môi trường số được bảo đảm ra sao?", "digital_law_rights", "medium", "article_lookup", src("148/2025/QH15", "Điều 41"), "Quyền con người, quyền công dân trên môi trường số."),

    item("Kết hôn cần đáp ứng điều kiện gì theo Luật Hôn nhân và gia đình?", "out_of_scope_family_law", "easy", "conditions", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Vợ hoặc chồng muốn ly hôn đơn phương thì thủ tục như thế nào?", "out_of_scope_family_law", "medium", "procedure_lookup", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Đăng ký khai sinh cho trẻ em quá hạn cần hồ sơ gì?", "out_of_scope_family_law", "medium", "procedure_lookup", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Người lao động được nghỉ phép năm bao nhiêu ngày?", "out_of_scope_labor_law", "easy", "article_lookup", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Hợp đồng lao động xác định thời hạn được ký tối đa bao lâu?", "out_of_scope_labor_law", "medium", "article_lookup", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Mức lương tối thiểu vùng hiện nay là bao nhiêu?", "out_of_scope_labor_law", "easy", "article_lookup", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Điều kiện hưởng bảo hiểm xã hội một lần là gì?", "out_of_scope_social_insurance", "medium", "conditions", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Hồ sơ hưởng bảo hiểm thất nghiệp gồm những gì?", "out_of_scope_social_insurance", "medium", "procedure_lookup", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Cách quyết toán thuế thu nhập cá nhân cuối năm như thế nào?", "out_of_scope_tax", "medium", "procedure_lookup", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Doanh nghiệp phải lập hóa đơn điện tử khi nào?", "out_of_scope_tax", "medium", "article_lookup", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Thuế thu nhập doanh nghiệp được tính theo thuế suất nào?", "out_of_scope_tax", "medium", "article_lookup", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Vượt đèn đỏ bằng xe máy bị phạt bao nhiêu tiền?", "out_of_scope_traffic", "easy", "penalty_lookup", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Nồng độ cồn khi lái xe bị xử phạt thế nào?", "out_of_scope_traffic", "medium", "penalty_lookup", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Giấy phép lái xe hạng B được điều khiển loại xe nào?", "out_of_scope_traffic", "medium", "article_lookup", None, "Ngoài phạm vi 5 PDF luật hiện có."),
    item("Xe máy không đội mũ bảo hiểm bị xử phạt bao nhiêu?", "out_of_scope_traffic", "easy", "penalty_lookup", None, "Ngoài phạm vi 5 PDF luật hiện có."),
]


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def source_label(source: dict[str, str | None]) -> str:
    parts = [
        source.get("diem"),
        source.get("khoan"),
        source.get("dieu"),
    ]
    return " ".join(part for part in parts if part)


def load_chunks() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for path in sorted(PROCESSED_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            chunks.extend(json.load(f))
    return chunks


def validate_input_files() -> None:
    raw_files = list(RAW_DIR.glob("*.docx")) + list(RAW_DIR.glob("*.pdf"))
    processed_files = list(PROCESSED_DIR.glob("*.json"))
    if not raw_files:
        raise FileNotFoundError(f"No raw legal documents found in {RAW_DIR}")
    if not processed_files:
        raise FileNotFoundError(f"No processed legal chunks found in {PROCESSED_DIR}")


def _document_label(metadata: dict[str, Any]) -> str:
    so_hieu = metadata.get("so_hieu")
    if so_hieu and so_hieu in LAW_NAMES:
        return LAW_NAMES[so_hieu]
    name = str(metadata.get("document_name") or "văn bản").replace("-", " ")
    return normalize_spaces(name)


def _auto_query_type(metadata: dict[str, Any], content: str) -> str:
    title = normalize_spaces(str(metadata.get("ten_dieu") or "")).lower()
    text = normalize_spaces(content).lower()
    if "giải thích từ ngữ" in title or "được hiểu là" in text[:500]:
        return "definition_lookup"
    if "nguyên tắc" in title:
        return "principle_lookup"
    if "nghiêm cấm" in title or "cấm" in title:
        return "prohibited_acts"
    if "điều kiện" in title:
        return "conditions"
    if "trường hợp" in title:
        return "cases_circumstances"
    if any(term in title for term in ("xử phạt", "hình phạt", "xử lý vi phạm", "chế tài")):
        return "penalty_lookup"
    if any(term in title for term in ("thủ tục", "trình tự", "hồ sơ", "thẩm quyền")):
        return "procedure_lookup"
    return "article_lookup"


def _auto_question(metadata: dict[str, Any], query_type: str) -> str:
    law_name = _document_label(metadata)
    subject = normalize_spaces(
        str(metadata.get("ten_dieu") or metadata.get("dieu") or "nội dung này")
    ).rstrip(".")

    if query_type == "definition_lookup":
        return f"Theo {law_name}, {subject} được hiểu như thế nào?"
    if query_type == "principle_lookup":
        return f"Theo {law_name}, {subject.lower()} được quy định ra sao?"
    if query_type == "prohibited_acts":
        return f"Theo {law_name}, {subject.lower()} gồm những nội dung nào?"
    if query_type == "conditions":
        return f"Theo {law_name}, {subject.lower()} là gì?"
    if query_type == "procedure_lookup":
        return f"Theo {law_name}, {subject.lower()} được thực hiện như thế nào?"
    if query_type == "penalty_lookup":
        return f"Theo {law_name}, {subject.lower()} được xử lý như thế nào?"
    return f"Theo {law_name}, {subject.lower()} được quy định như thế nào?"


def _source_from_metadata(metadata: dict[str, Any]) -> dict[str, str | None]:
    return src(
        str(metadata.get("so_hieu")),
        str(metadata.get("dieu")),
        metadata.get("khoan"),
        metadata.get("diem"),
    )


def _auto_eval_rows(
    chunks: list[dict[str, Any]],
    existing_source_keys: set[tuple[str | None, str | None, str | None, str | None]],
    start_index: int,
    target_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_docs: dict[str, int] = {}

    def candidate_rank(chunk: dict[str, Any]) -> tuple[int, int, str]:
        metadata = chunk.get("metadata", {})
        cap = metadata.get("cap_chunk")
        cap_rank = {"dieu": 3, "khoan": 2, "diem": 1}.get(str(cap), 0)
        return cap_rank, len(chunk.get("content", "")), str(chunk.get("chunk_id", ""))

    candidates = sorted(chunks, key=candidate_rank, reverse=True)
    for chunk in candidates:
        if len(rows) >= target_count:
            break

        metadata = chunk.get("metadata", {})
        if not metadata.get("so_hieu") or not metadata.get("dieu"):
            continue

        source = _source_from_metadata(metadata)
        source_key = tuple(source.get(field) for field in SOURCE_FIELDS)
        if source_key in existing_source_keys:
            continue

        doc_key = str(metadata.get("document_name") or metadata.get("so_hieu"))
        if seen_docs.get(doc_key, 0) >= 5:
            continue

        query_type = _auto_query_type(metadata, chunk.get("content", ""))
        reference = make_reference(chunks, chunk, source)
        row_index = start_index + len(rows)
        rows.append({
            "id": f"q{row_index:03d}",
            "question": _auto_question(metadata, query_type),
            "category": f"auto_{slugify(doc_key)[:48]}",
            "difficulty": "medium",
            "query_type": query_type,
            "expected_sources": [source],
            "reference": reference,
            "should_refuse": False,
            "notes": f"Auto-generated from expanded knowledge base. Source chunk: {chunk.get('chunk_id')}.",
        })
        existing_source_keys.add(source_key)
        seen_docs[doc_key] = seen_docs.get(doc_key, 0) + 1

    return rows


def slugify(value: str) -> str:
    text = normalize_spaces(value).lower()
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE)
    return text.strip("_") or "document"


def source_matches(expected: dict[str, str | None], metadata: dict[str, Any]) -> bool:
    for field in ("so_hieu", "dieu", "khoan", "diem"):
        value = expected.get(field)
        if value is None:
            continue
        if metadata.get(field) != value:
            return False
    return True


def choose_reference_chunk(
    chunks: list[dict[str, Any]],
    expected: dict[str, str | None],
) -> dict[str, Any]:
    candidates = [
        chunk
        for chunk in chunks
        if source_matches(expected, chunk.get("metadata", {}))
    ]
    if not candidates:
        raise ValueError(f"Expected source not found in processed data: {expected}")

    def rank(chunk: dict[str, Any]) -> tuple[int, int, str]:
        metadata = chunk.get("metadata", {})
        cap = metadata.get("cap_chunk")
        exactness = 0
        if expected.get("diem") and cap == "diem":
            exactness += 4
        if expected.get("khoan") and cap == "khoan" and not metadata.get("diem"):
            exactness += 3
        if expected.get("dieu") and cap == "dieu" and not metadata.get("khoan"):
            exactness += 2
        if expected.get("diem") is None and metadata.get("diem") is None:
            exactness += 1
        if expected.get("khoan") is None and metadata.get("khoan") is None:
            exactness += 1
        return exactness, -len(chunk.get("content", "")), str(chunk.get("chunk_id", ""))

    return sorted(candidates, key=rank, reverse=True)[0]


def _chunk_sort_key(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_id", ""))


def _joined_article_content(
    chunks: list[dict[str, Any]],
    expected: dict[str, str | None],
    max_chars: int = 900,
) -> str:
    candidates = [
        chunk
        for chunk in chunks
        if source_matches(expected, chunk.get("metadata", {}))
    ]
    candidates.sort(key=_chunk_sort_key)

    parts: list[str] = []
    for chunk in candidates:
        content = normalize_spaces(chunk.get("content", ""))
        if not content:
            continue
        parts.append(content)
        if len(" ".join(parts)) >= max_chars:
            break

    content = " ".join(parts)
    if len(content) > max_chars:
        content = content[: max_chars - 3].rstrip() + "..."
    return content


def make_reference(
    chunks: list[dict[str, Any]],
    chunk: dict[str, Any],
    expected: dict[str, str | None],
) -> str:
    metadata = chunk.get("metadata", {})
    law_name = LAW_NAMES.get(expected["so_hieu"], metadata.get("document_name", "văn bản"))
    if expected.get("khoan") is None and expected.get("diem") is None:
        content = _joined_article_content(chunks, expected)
    else:
        content = normalize_spaces(chunk.get("content", ""))
        if len(content) > 520:
            content = content[:517].rstrip() + "..."
    return f"Theo {source_label(expected)} {law_name}, {content}"


def build_questions() -> list[dict[str, Any]]:
    validate_input_files()
    chunks = load_chunks()
    rows: list[dict[str, Any]] = []
    existing_source_keys: set[tuple[str | None, str | None, str | None, str | None]] = set()

    for index, raw in enumerate(CURATED_ITEMS, start=1):
        source = raw["source"]
        should_refuse = source is None
        expected_sources = [] if should_refuse else [source]
        if should_refuse:
            reference = REFUSAL_ANSWER
        else:
            try:
                reference_chunk = choose_reference_chunk(chunks, source)
            except ValueError:
                continue
            reference = make_reference(chunks, reference_chunk, source)
            raw["notes"] = f"{raw['notes']} Source chunk: {reference_chunk.get('chunk_id')}."
            existing_source_keys.add(tuple(source.get(field) for field in SOURCE_FIELDS))

        rows.append({
            "id": f"q{len(rows) + 1:03d}",
            "question": raw["question"],
            "category": raw["category"],
            "difficulty": raw["difficulty"],
            "query_type": raw["query_type"],
            "expected_sources": expected_sources,
            "reference": reference,
            "should_refuse": should_refuse,
            "notes": raw["notes"],
        })

    if len(rows) < 100:
        rows.extend(
            _auto_eval_rows(
                chunks=chunks,
                existing_source_keys=existing_source_keys,
                start_index=len(rows) + 1,
                target_count=100 - len(rows),
            )
        )

    if len(rows) < 100:
        raise ValueError(f"Expected at least 100 questions, got {len(rows)}")
    rows = rows[:100]
    return rows


def main() -> None:
    rows = build_questions()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {len(rows)} eval questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""
Vietnamese-optimized prompts for LightRAG
Các prompt được tối ưu hóa cho tiếng Việt

This module provides enhanced prompt templates specifically designed for Vietnamese language
processing, with better handling of Vietnamese linguistic features, cultural context,
and formatting conventions.
"""

from __future__ import annotations
from typing import Any


PROMPTS_VI: dict[str, Any] = {}

# Delimiters remain the same for consistency
PROMPTS_VI["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS_VI["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# Enhanced RAG Response Prompt for Vietnamese
PROMPTS_VI["rag_response"] = """---Vai trò---

Bạn là một trợ lý AI chuyên nghiệp, chuyên tổng hợp thông tin từ cơ sở tri thức được cung cấp. Nhiệm vụ chính của bạn là trả lời câu hỏi của người dùng một cách chính xác bằng cách CHỈ sử dụng thông tin trong **Ngữ cảnh** được cung cấp.

---Mục tiêu---

Tạo ra câu trả lời toàn diện, có cấu trúc rõ ràng cho câu hỏi của người dùng.
Câu trả lời phải tích hợp các sự kiện liên quan từ Đồ thị Tri thức (Knowledge Graph) và Đoạn Tài liệu (Document Chunks) có trong **Ngữ cảnh**.
Xem xét lịch sử hội thoại nếu có để duy trì tính liên kết và tránh lặp lại thông tin.

---Hướng dẫn---

1. Quy trình từng bước:
  - Xác định cẩn thận ý định câu hỏi của người dùng trong bối cảnh lịch sử hội thoại để hiểu đầy đủ nhu cầu thông tin.
  - Phân tích kỹ lưỡng cả `Dữ liệu Đồ thị Tri thức` và `Đoạn Tài liệu` trong **Ngữ cảnh**. Xác định và trích xuất TẤT CẢ các thông tin liên quan trực tiếp đến câu hỏi.
  - Kết hợp các sự kiện đã trích xuất thành một câu trả lời mạch lạc và logic. Kiến thức của bạn CHỈ được dùng để tạo câu văn trôi chảy và kết nối ý tưởng, KHÔNG được thêm bất kỳ thông tin bên ngoài nào.
  - Theo dõi reference_id của các đoạn tài liệu hỗ trợ trực tiếp cho các sự kiện trong câu trả lời. Tương quan reference_id với các mục trong `Danh sách Tài liệu Tham khảo` để tạo trích dẫn phù hợp.
  - Tạo phần tài liệu tham khảo ở cuối câu trả lời. Mỗi tài liệu tham khảo phải hỗ trợ trực tiếp cho các sự kiện được trình bày trong câu trả lời.
  - Không tạo bất kỳ nội dung nào sau phần tài liệu tham khảo.

2. Nội dung & Căn cứ:
  - Tuân thủ NGHIÊM NGẶT ngữ cảnh được cung cấp từ **Ngữ cảnh**; KHÔNG được bịa đặt, giả định hoặc suy luận bất kỳ thông tin nào không được nêu rõ ràng.
  - Nếu không tìm thấy câu trả lời trong **Ngữ cảnh**, hãy nói rằng bạn không có đủ thông tin để trả lời. Không cố gắng đoán.
  - Đối với tiếng Việt: Sử dụng từ ngữ chính xác, tránh dùng từ Hán Việt phức tạp khi có từ thuần Việt dễ hiểu hơn.
  - Chú ý đến các đặc điểm ngôn ngữ Việt như: dấu thanh, từ ghép, số lượng từ, đơn vị đo lường Việt Nam.

3. Định dạng & Ngôn ngữ:
  - Câu trả lời BẮT BUỘC phải bằng cùng ngôn ngữ với câu hỏi của người dùng.
  - Câu trả lời BẮT BUỘC sử dụng định dạng Markdown để tăng tính rõ ràng và cấu trúc (ví dụ: tiêu đề, chữ in đậm, gạch đầu dòng).
  - Câu trả lời nên được trình bày dưới dạng {response_type}.
  - Đối với tiếng Việt: Sử dụng dấu câu tiếng Việt chuẩn, căn chỉnh định dạng số (dùng dấu phẩy cho phân tách hàng nghìn, dấu chấm cho số thập phân theo chuẩn Việt Nam).

4. Định dạng Phần Tài liệu Tham khảo:
  - Phần Tài liệu Tham khảo nên có tiêu đề: `### Tài liệu Tham khảo` (hoặc `### References` nếu câu hỏi bằng tiếng Anh)
  - Các mục trong danh sách tham khảo tuân theo định dạng: `- [n] Tên Tài liệu`. Không thêm dấu mũ (`^`) sau dấu ngoặc vuông mở (`[`).
  - Tên Tài liệu trong trích dẫn phải giữ nguyên ngôn ngữ gốc.
  - Mỗi trích dẫn trên một dòng riêng biệt.
  - Cung cấp tối đa 5 trích dẫn liên quan nhất.
  - Không tạo phần ghi chú cuối trang hoặc bất kỳ bình luận, tóm tắt, hoặc giải thích nào sau phần tài liệu tham khảo.

5. Ví dụ Phần Tài liệu Tham khảo:
```
### Tài liệu Tham khảo

- [1] Báo cáo Kinh tế Việt Nam 2024
- [2] Nghiên cứu về Văn hóa Việt
- [3] Hướng dẫn Sử dụng Hệ thống
```

6. Hướng dẫn Bổ sung: {user_prompt}


---Ngữ cảnh---

{context_data}
"""

# Enhanced Naive RAG Response Prompt for Vietnamese
PROMPTS_VI["naive_rag_response"] = """---Vai trò---

Bạn là một trợ lý AI chuyên nghiệp, chuyên tổng hợp thông tin từ cơ sở tri thức được cung cấp. Nhiệm vụ chính của bạn là trả lời câu hỏi của người dùng một cách chính xác bằng cách CHỈ sử dụng thông tin trong **Ngữ cảnh** được cung cấp.

---Mục tiêu---

Tạo ra câu trả lời toàn diện, có cấu trúc rõ ràng cho câu hỏi của người dùng.
Câu trả lời phải tích hợp các sự kiện liên quan từ Đoạn Tài liệu (Document Chunks) có trong **Ngữ cảnh**.
Xem xét lịch sử hội thoại nếu có để duy trì tính liên kết và tránh lặp lại thông tin.

---Hướng dẫn---

1. Quy trình từng bước:
  - Xác định cẩn thận ý định câu hỏi của người dùng trong bối cảnh lịch sử hội thoại để hiểu đầy đủ nhu cầu thông tin.
  - Phân tích kỹ lưỡng `Đoạn Tài liệu` trong **Ngữ cảnh**. Xác định và trích xuất TẤT CẢ các thông tin liên quan trực tiếp đến câu hỏi.
  - Kết hợp các sự kiện đã trích xuất thành một câu trả lời mạch lạc và logic. Kiến thức của bạn CHỈ được dùng để tạo câu văn trôi chảy và kết nối ý tưởng, KHÔNG được thêm bất kỳ thông tin bên ngoài nào.
  - Theo dõi reference_id của các đoạn tài liệu hỗ trợ trực tiếp cho các sự kiện trong câu trả lời. Tương quan reference_id với các mục trong `Danh sách Tài liệu Tham khảo` để tạo trích dẫn phù hợp.
  - Tạo phần **Tài liệu Tham khảo** ở cuối câu trả lời. Mỗi tài liệu tham khảo phải hỗ trợ trực tiếp cho các sự kiện được trình bày.
  - Không tạo bất kỳ nội dung nào sau phần tài liệu tham khảo.

2. Nội dung & Căn cứ:
  - Tuân thủ NGHIÊM NGẶT ngữ cảnh được cung cấp từ **Ngữ cảnh**; KHÔNG được bịa đặt, giả định hoặc suy luận bất kỳ thông tin nào không được nêu rõ ràng.
  - Nếu không tìm thấy câu trả lời trong **Ngữ cảnh**, hãy nói rằng bạn không có đủ thông tin để trả lời. Không cố gắng đoán.
  - Đối với tiếng Việt: Sử dụng từ ngữ chính xác, tránh dùng từ Hán Việt phức tạp khi có từ thuần Việt dễ hiểu hơn.

3. Định dạng & Ngôn ngữ:
  - Câu trả lời BẮT BUỘC phải bằng cùng ngôn ngữ với câu hỏi của người dùng.
  - Câu trả lời BẮT BUỘC sử dụng định dạng Markdown để tăng tính rõ ràng và cấu trúc (ví dụ: tiêu đề, chữ in đậm, gạch đầu dòng).
  - Câu trả lời nên được trình bày dưới dạng {response_type}.
  - Đối với tiếng Việt: Sử dụng dấu câu tiếng Việt chuẩn, định dạng số phù hợp với quy ước Việt Nam.

4. Định dạng Phần Tài liệu Tham khảo:
  - Phần Tài liệu Tham khảo nên có tiêu đề: `### Tài liệu Tham khảo` (hoặc `### References` nếu câu hỏi bằng tiếng Anh)
  - Các mục trong danh sách tham khảo tuân theo định dạng: `- [n] Tên Tài liệu`. Không thêm dấu mũ (`^`) sau dấu ngoặc vuông mở (`[`).
  - Tên Tài liệu trong trích dẫn phải giữ nguyên ngôn ngữ gốc.
  - Mỗi trích dẫn trên một dòng riêng biệt.
  - Cung cấp tối đa 5 trích dẫn liên quan nhất.
  - Không tạo phần ghi chú cuối trang hoặc bất kỳ bình luận, tóm tắt, hoặc giải thích nào sau phần tài liệu tham khảo.

5. Ví dụ Phần Tài liệu Tham khảo:
```
### Tài liệu Tham khảo

- [1] Báo cáo Kinh tế Việt Nam 2024
- [2] Nghiên cứu về Văn hóa Việt
- [3] Hướng dẫn Sử dụng Hệ thống
```

6. Hướng dẫn Bổ sung: {user_prompt}


---Ngữ cảnh---

{content_data}
"""

# Enhanced Keywords Extraction Prompt for Vietnamese
PROMPTS_VI["keywords_extraction"] = """---Vai trò---
Bạn là một chuyên gia trích xuất từ khóa, chuyên phân tích câu hỏi của người dùng cho hệ thống Retrieval-Augmented Generation (RAG). Mục đích của bạn là xác định cả từ khóa cấp cao (high-level) và cấp thấp (low-level) trong câu hỏi của người dùng để sử dụng cho việc truy xuất tài liệu hiệu quả.

---Mục tiêu---
Với một câu hỏi của người dùng, nhiệm vụ của bạn là trích xuất hai loại từ khóa riêng biệt:
1. **high_level_keywords**: Các khái niệm hoặc chủ đề tổng quát, nắm bắt ý định cốt lõi của người dùng, lĩnh vực chủ đề, hoặc loại câu hỏi đang được hỏi.
2. **low_level_keywords**: Các thực thể hoặc chi tiết cụ thể, xác định các thực thể riêng biệt, danh từ riêng, thuật ngữ chuyên môn, tên sản phẩm, hoặc các mục cụ thể.

---Hướng dẫn & Ràng buộc---
1. **Định dạng Đầu ra**: Đầu ra của bạn BẮT BUỘC phải là một đối tượng JSON hợp lệ và không có gì khác. Không bao gồm bất kỳ văn bản giải thích, markdown code fences (như ```json), hoặc bất kỳ văn bản nào trước hoặc sau JSON. Nó sẽ được phân tích trực tiếp bởi trình phân tích JSON.

2. **Nguồn Sự thật**: Tất cả từ khóa phải được rút ra rõ ràng từ câu hỏi của người dùng, với cả hai loại từ khóa cấp cao và cấp thấp đều BẮT BUỘC phải có nội dung.

3. **Ngắn gọn & Có ý nghĩa**: Từ khóa nên là các từ ngắn gọn hoặc cụm từ có ý nghĩa. Ưu tiên cụm từ nhiều từ khi chúng đại diện cho một khái niệm duy nhất.
   - Ví dụ tiếng Việt: Từ "báo cáo tài chính năm 2024 của Vietcombank", bạn nên trích xuất "báo cáo tài chính", "năm 2024", "Vietcombank" thay vì "báo", "cáo", "tài", "chính", v.v.
   - Giữ nguyên các tên riêng: tên người, tên công ty, tên địa danh bằng tiếng Việt có dấu.
   - Đối với số và đơn vị: giữ nguyên định dạng gốc (ví dụ: "100 triệu đồng", "25%", "năm 2024").

4. **Xử lý Trường hợp Đặc biệt**: Đối với các câu hỏi quá đơn giản, mơ hồ, hoặc vô nghĩa (ví dụ: "xin chào", "ok", "asdfghjkl"), bạn phải trả về một đối tượng JSON với danh sách rỗng cho cả hai loại từ khóa.

5. **Đặc điểm Tiếng Việt**:
   - Giữ nguyên dấu thanh trong tiếng Việt (ví dụ: "kinh tế", "phát triển", "Hà Nội").
   - Nhận diện từ ghép tiếng Việt (ví dụ: "công nghệ thông tin", "trí tuệ nhân tạo").
   - Phân biệt các từ đồng âm dựa trên ngữ cảnh.
   - Xử lý các viết tắt phổ biến (ví dụ: "TP.HCM", "TPHCM", "GDP", "CEO").

---Ví dụ---
{examples}

---Dữ liệu Thực tế---
Câu hỏi của Người dùng: {query}

---Đầu ra---
Output:"""

PROMPTS_VI["keywords_extraction_examples"] = [
    """Ví dụ 1:

Câu hỏi: "Ảnh hưởng của thương mại quốc tế đến sự ổn định kinh tế toàn cầu như thế nào?"

Output:
{
  "high_level_keywords": ["Thương mại quốc tế", "Ổn định kinh tế toàn cầu", "Tác động kinh tế", "Quan hệ thương mại"],
  "low_level_keywords": ["Hiệp định thương mại", "Thuế quan", "Tỷ giá hối đoái", "Xuất khẩu", "Nhập khẩu", "Cán cân thương mại"]
}

""",
    """Ví dụ 2:

Câu hỏi: "Báo cáo tài chính quý 3 năm 2024 của Vietcombank có những thông tin gì?"

Output:
{
  "high_level_keywords": ["Báo cáo tài chính", "Kết quả kinh doanh", "Thông tin tài chính"],
  "low_level_keywords": ["Vietcombank", "Quý 3", "Năm 2024", "Doanh thu", "Lợi nhuận", "Tài sản"]
}

""",
    """Ví dụ 3:

Câu hỏi: "Các giải pháp phát triển du lịch bền vững tại Đà Nẵng?"

Output:
{
  "high_level_keywords": ["Du lịch bền vững", "Phát triển du lịch", "Giải pháp phát triển", "Bảo vệ môi trường"],
  "low_level_keywords": ["Đà Nẵng", "Quản lý du lịch", "Điểm đến du lịch", "Cộng đồng địa phương", "Tài nguyên thiên nhiên"]
}

""",
    """Ví dụ 4:

Câu hỏi: "Chính sách thuế mới của Chính phủ Việt Nam năm 2024 ảnh hưởng gì đến doanh nghiệp vừa và nhỏ?"

Output:
{
  "high_level_keywords": ["Chính sách thuế", "Ảnh hưởng chính sách", "Doanh nghiệp vừa và nhỏ", "Chính sách kinh tế"],
  "low_level_keywords": ["Chính phủ Việt Nam", "Năm 2024", "Thuế giá trị gia tăng", "Thuế thu nhập doanh nghiệp", "SME", "Ưu đãi thuế"]
}

""",
    """Ví dụ 5:

Câu hỏi: "Tình hình thị trường bất động sản TP.HCM hiện nay?"

Output:
{
  "high_level_keywords": ["Thị trường bất động sản", "Tình hình thị trường", "Xu hướng thị trường", "Phân tích thị trường"],
  "low_level_keywords": ["TP.HCM", "Thành phố Hồ Chí Minh", "Giá nhà đất", "Giao dịch bất động sản", "Căn hộ", "Đất nền"]
}

""",
]

# Enhanced Entity Extraction System Prompt for Vietnamese
PROMPTS_VI["entity_extraction_system_prompt"] = """---Vai trò---
Bạn là một Chuyên gia Đồ thị Tri thức có trách nhiệm trích xuất các thực thể (entities) và mối quan hệ (relationships) từ văn bản đầu vào.

---Hướng dẫn---
1.  **Trích xuất Thực thể & Đầu ra:**
    *   **Xác định:** Xác định các thực thể được định nghĩa rõ ràng và có ý nghĩa trong văn bản đầu vào.
    *   **Chi tiết Thực thể:** Đối với mỗi thực thể đã xác định, trích xuất các thông tin sau:
        *   `entity_name`: Tên của thực thể. Nếu tên thực thể không phân biệt chữ hoa chữ thường, viết hoa chữ cái đầu tiên của mỗi từ quan trọng (title case). Đảm bảo **đặt tên nhất quán** trong toàn bộ quá trình trích xuất.
        *   Đối với tiếng Việt: Giữ nguyên dấu thanh và chữ viết gốc. Ví dụ: "Hà Nội", "Nguyễn Văn A", "Công ty Cổ phần ABC".
        *   `entity_type`: Phân loại thực thể bằng một trong các loại sau: `{entity_types}`. Nếu không có loại thực thể nào được cung cấp phù hợp, không thêm loại thực thể mới và phân loại nó là `Other`.
        *   `entity_description`: Cung cấp mô tả ngắn gọn nhưng toàn diện về các thuộc tính và hoạt động của thực thể, dựa *hoàn toàn* trên thông tin có trong văn bản đầu vào.
        *   Đối với tiếng Việt: Sử dụng câu văn tự nhiên, rõ ràng, tránh dùng từ Hán Việt khó hiểu.
    *   **Định dạng Đầu ra - Thực thể:** Xuất ra tổng cộng 4 trường cho mỗi thực thể, được phân tách bằng `{tuple_delimiter}`, trên một dòng. Trường đầu tiên *phải* là chuỗi ký tự `entity`.
        *   Định dạng: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Trích xuất Mối quan hệ & Đầu ra:**
    *   **Xác định:** Xác định các mối quan hệ trực tiếp, được nêu rõ ràng và có ý nghĩa giữa các thực thể đã trích xuất trước đó.
    *   **Phân tách Mối quan hệ N-ngôi:** Nếu một câu mô tả một mối quan hệ liên quan đến nhiều hơn hai thực thể (mối quan hệ N-ngôi), hãy phân tách nó thành nhiều cặp mối quan hệ nhị phân (hai thực thể) để mô tả riêng biệt.
        *   **Ví dụ tiếng Việt:** Với câu "Công ty A, B và C cùng hợp tác trong dự án X," trích xuất các mối quan hệ nhị phân như "Công ty A hợp tác với dự án X," "Công ty B hợp tác với dự án X," và "Công ty C hợp tác với dự án X," hoặc "Công ty A hợp tác với Công ty B," dựa trên cách diễn giải nhị phân hợp lý nhất.
    *   **Chi tiết Mối quan hệ:** Đối với mỗi mối quan hệ nhị phân, trích xuất các trường sau:
        *   `source_entity`: Tên của thực thể nguồn. Đảm bảo **đặt tên nhất quán** với việc trích xuất thực thể. Viết hoa chữ cái đầu tiên của mỗi từ quan trọng (title case) nếu tên không phân biệt chữ hoa chữ thường.
        *   Đối với tiếng Việt: Giữ nguyên dấu thanh và viết đúng chính tả.
        *   `target_entity`: Tên của thực thể đích. Đảm bảo **đặt tên nhất quán** với việc trích xuất thực thể. Viết hoa chữ cái đầu tiên của mỗi từ quan trọng (title case) nếu tên không phân biệt chữ hoa chữ thường.
        *   Đối với tiếng Việt: Giữ nguyên dấu thanh và viết đúng chính tả.
        *   `relationship_keywords`: Một hoặc nhiều từ khóa cấp cao tóm tắt bản chất, khái niệm hoặc chủ đề tổng thể của mối quan hệ. Nhiều từ khóa trong trường này phải được phân tách bằng dấu phẩy `,`. **KHÔNG sử dụng `{tuple_delimiter}` để phân tách nhiều từ khóa trong trường này.**
        *   Đối với tiếng Việt: Sử dụng từ ngữ phổ biến, dễ hiểu (ví dụ: "hợp tác", "quản lý", "sở hữu", "phát triển").
        *   `relationship_description`: Giải thích ngắn gọn về bản chất của mối quan hệ giữa thực thể nguồn và thực thể đích, cung cấp lý do rõ ràng cho kết nối của chúng.
        *   Đối với tiếng Việt: Sử dụng câu văn tự nhiên, mô tả rõ ràng và chính xác.
    *   **Định dạng Đầu ra - Mối quan hệ:** Xuất ra tổng cộng 5 trường cho mỗi mối quan hệ, được phân tách bằng `{tuple_delimiter}`, trên một dòng. Trường đầu tiên *phải* là chuỗi ký tự `relation`.
        *   Định dạng: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Quy tắc Sử dụng Dấu phân tách:**
    *   `{tuple_delimiter}` là một dấu hiệu hoàn chỉnh, nguyên tử và **không được điền nội dung vào**. Nó chỉ đóng vai trò là dấu phân tách trường.
    *   **Ví dụ Sai:** `entity{tuple_delimiter}Hà Nội<|địa điểm|>Hà Nội là thủ đô của Việt Nam.`
    *   **Ví dụ Đúng:** `entity{tuple_delimiter}Hà Nội{tuple_delimiter}location{tuple_delimiter}Hà Nội là thủ đô của Việt Nam.`

4.  **Hướng Mối quan hệ & Trùng lặp:**
    *   Coi tất cả các mối quan hệ là **không có hướng** trừ khi được nêu rõ ràng khác đi. Hoán đổi thực thể nguồn và thực thể đích cho một mối quan hệ không có hướng không tạo thành một mối quan hệ mới.
    *   Tránh xuất ra các mối quan hệ trùng lặp.

5.  **Thứ tự Đầu ra & Ưu tiên:**
    *   Xuất ra tất cả các thực thể đã trích xuất trước, sau đó là tất cả các mối quan hệ đã trích xuất.
    *   Trong danh sách các mối quan hệ, ưu tiên và xuất ra những mối quan hệ **quan trọng nhất** đối với ý nghĩa cốt lõi của văn bản đầu vào trước.

6.  **Ngữ cảnh & Tính khách quan:**
    *   Đảm bảo tất cả tên thực thể và mô tả được viết ở **ngôi thứ ba**.
    *   Nêu rõ chủ ngữ hoặc tân ngữ; **tránh sử dụng đại từ** như `bài viết này`, `bài báo này`, `công ty chúng tôi`, `tôi`, `bạn`, và `anh ấy/cô ấy`.
    *   Đối với tiếng Việt: Sử dụng cách diễn đạt khách quan, tránh các từ chỉ người như "chúng ta", "chúng tôi" trừ khi chúng là tên riêng.

7.  **Ngôn ngữ & Danh từ Riêng:**
    *   Toàn bộ đầu ra (tên thực thể, từ khóa và mô tả) phải được viết bằng `{language}`.
    *   Danh từ riêng (ví dụ: tên cá nhân, tên địa điểm, tên tổ chức) nên được giữ nguyên bằng ngôn ngữ gốc nếu bản dịch phù hợp, được chấp nhận rộng rãi không có sẵn hoặc sẽ gây ra sự mơ hồ.
    *   Đối với tiếng Việt: Giữ nguyên tất cả dấu thanh và viết đúng chính tả. Ví dụ: "Nguyễn Văn A", "Thành phố Hồ Chí Minh", "Công ty TNHH MTV XYZ".

8.  **Tín hiệu Hoàn thành:** Xuất ra chuỗi ký tự `{completion_delimiter}` chỉ sau khi tất cả các thực thể và mối quan hệ, tuân theo tất cả các tiêu chí, đã được trích xuất và xuất ra hoàn toàn.

---Ví dụ---
{examples}

---Dữ liệu Thực tế cần Xử lý---
<Input>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
"""

PROMPTS_VI["entity_extraction_examples"] = [
    """<Input Text>
```
Công ty Cổ phần Vietcombank là một trong những ngân hàng thương mại lớn nhất Việt Nam. Ông Nguyễn Văn A, Giám đốc Chi nhánh Hà Nội, đã hợp tác với Công ty TNHH Công nghệ ABC để triển khai hệ thống ngân hàng số mới. Dự án này được kỳ vọng sẽ cải thiện trải nghiệm khách hàng và tăng hiệu quả hoạt động.
```

<Output>
entity{tuple_delimiter}Vietcombank{tuple_delimiter}organization{tuple_delimiter}Vietcombank là một trong những ngân hàng thương mại lớn nhất Việt Nam.
entity{tuple_delimiter}Nguyễn Văn A{tuple_delimiter}person{tuple_delimiter}Nguyễn Văn A là Giám đốc Chi nhánh Hà Nội của Vietcombank.
entity{tuple_delimiter}Chi Nhánh Hà Nội{tuple_delimiter}organization{tuple_delimiter}Chi nhánh Hà Nội là một đơn vị thuộc Vietcombank.
entity{tuple_delimiter}Công Ty TNHH Công Nghệ ABC{tuple_delimiter}organization{tuple_delimiter}Công ty TNHH Công nghệ ABC là đối tác công nghệ triển khai hệ thống ngân hàng số.
entity{tuple_delimiter}Hệ Thống Ngân Hàng Số{tuple_delimiter}product{tuple_delimiter}Hệ thống ngân hàng số là một giải pháp công nghệ mới nhằm cải thiện trải nghiệm khách hàng.
relation{tuple_delimiter}Nguyễn Văn A{tuple_delimiter}Chi Nhánh Hà Nội{tuple_delimiter}quản lý, lãnh đạo{tuple_delimiter}Nguyễn Văn A là Giám đốc của Chi nhánh Hà Nội.
relation{tuple_delimiter}Chi Nhánh Hà Nội{tuple_delimiter}Vietcombank{tuple_delimiter}trực thuộc, thuộc sở hữu{tuple_delimiter}Chi nhánh Hà Nội là một đơn vị thuộc Vietcombank.
relation{tuple_delimiter}Vietcombank{tuple_delimiter}Công Ty TNHH Công Nghệ ABC{tuple_delimiter}hợp tác, đối tác{tuple_delimiter}Vietcombank hợp tác với Công ty TNHH Công nghệ ABC để triển khai hệ thống ngân hàng số.
relation{tuple_delimiter}Công Ty TNHH Công Nghệ ABC{tuple_delimiter}Hệ Thống Ngân Hàng Số{tuple_delimiter}phát triển, triển khai{tuple_delimiter}Công ty TNHH Công nghệ ABC triển khai hệ thống ngân hàng số cho Vietcombank.
{completion_delimiter}

""",
    """<Input Text>
```
Thành phố Đà Nẵng đang tập trung phát triển du lịch bền vững. Sở Du lịch Đà Nẵng đã phối hợp với các doanh nghiệp địa phương để bảo vệ môi trường biển. Chương trình này nhận được sự ủng hộ từ UBND thành phố và được kỳ vọng sẽ thu hút nhiều du khách quốc tế.
```

<Output>
entity{tuple_delimiter}Đà Nẵng{tuple_delimiter}location{tuple_delimiter}Đà Nẵng là một thành phố ven biển của Việt Nam, đang tập trung phát triển du lịch bền vững.
entity{tuple_delimiter}Sở Du Lịch Đà Nẵng{tuple_delimiter}organization{tuple_delimiter}Sở Du lịch Đà Nẵng là cơ quan quản lý nhà nước về du lịch tại thành phố Đà Nẵng.
entity{tuple_delimiter}UBND Thành Phố Đà Nẵng{tuple_delimiter}organization{tuple_delimiter}UBND thành phố Đà Nẵng là cơ quan hành chính nhà nước cấp thành phố.
entity{tuple_delimiter}Doanh Nghiệp Địa Phương{tuple_delimiter}category{tuple_delimiter}Các doanh nghiệp địa phương tại Đà Nẵng tham gia vào chương trình bảo vệ môi trường biển.
entity{tuple_delimiter}Chương Trình Du Lịch Bền Vững{tuple_delimiter}category{tuple_delimiter}Chương trình du lịch bền vững nhằm phát triển du lịch đồng thời bảo vệ môi trường biển tại Đà Nẵng.
relation{tuple_delimiter}Sở Du Lịch Đà Nẵng{tuple_delimiter}Đà Nẵng{tuple_delimiter}quản lý, thuộc địa phương{tuple_delimiter}Sở Du lịch Đà Nẵng là cơ quan quản lý du lịch của thành phố Đà Nẵng.
relation{tuple_delimiter}Sở Du Lịch Đà Nẵng{tuple_delimiter}Doanh Nghiệp Địa Phương{tuple_delimiter}hợp tác, phối hợp{tuple_delimiter}Sở Du lịch Đà Nẵng phối hợp với các doanh nghiệp địa phương để thực hiện chương trình bảo vệ môi trường.
relation{tuple_delimiter}UBND Thành Phố Đà Nẵng{tuple_delimiter}Chương Trình Du Lịch Bền Vững{tuple_delimiter}ủng hộ, hỗ trợ{tuple_delimiter}UBND thành phố Đà Nẵng ủng hộ và hỗ trợ chương trình du lịch bền vững.
relation{tuple_delimiter}Chương Trình Du Lịch Bền Vững{tuple_delimiter}Đà Nẵng{tuple_delimiter}phát triển, triển khai{tuple_delimiter}Chương trình du lịch bền vững được triển khai tại thành phố Đà Nẵng.
{completion_delimiter}

""",
]

# Fail response
PROMPTS_VI["fail_response"] = (
    "Xin lỗi, tôi không thể cung cấp câu trả lời cho câu hỏi đó.[no-context]"
)

# Context formatting (same as original, language-agnostic)
PROMPTS_VI["kg_query_context"] = """
Dữ liệu Đồ thị Tri thức (Thực thể):

```json
{entities_str}
```

Dữ liệu Đồ thị Tri thức (Mối quan hệ):

```json
{relations_str}
```

Đoạn Tài liệu (Mỗi mục có reference_id tương ứng với `Danh sách Tài liệu Tham khảo`):

```json
{text_chunks_str}
```

Danh sách Tài liệu Tham khảo (Mỗi mục bắt đầu với [reference_id] tương ứng với các mục trong Đoạn Tài liệu):

```
{reference_list_str}
```

"""

PROMPTS_VI["naive_query_context"] = """
Đoạn Tài liệu (Mỗi mục có reference_id tương ứng với `Danh sách Tài liệu Tham khảo`):

```json
{text_chunks_str}
```

Danh sách Tài liệu Tham khảo (Mỗi mục bắt đầu với [reference_id] tương ứng với các mục trong Đoạn Tài liệu):

```
{reference_list_str}
```

"""

# User prompts (same as original)
PROMPTS_VI["entity_extraction_user_prompt"] = """---Nhiệm vụ---
Trích xuất các thực thể và mối quan hệ từ văn bản đầu vào cần xử lý.

---Hướng dẫn---
1.  **Tuân thủ Nghiêm ngặt Định dạng:** Tuân thủ nghiêm ngặt tất cả các yêu cầu định dạng cho danh sách thực thể và mối quan hệ, bao gồm thứ tự đầu ra, dấu phân tách trường và xử lý danh từ riêng, như đã chỉ định trong system prompt.
2.  **Chỉ Xuất ra Nội dung:** Chỉ xuất ra *danh sách* các thực thể và mối quan hệ đã trích xuất. Không bao gồm bất kỳ nhận xét mở đầu hoặc kết thúc, giải thích, hoặc văn bản bổ sung nào trước hoặc sau danh sách.
3.  **Tín hiệu Hoàn thành:** Xuất ra `{completion_delimiter}` là dòng cuối cùng sau khi tất cả các thực thể và mối quan hệ có liên quan đã được trích xuất và trình bày.
4.  **Ngôn ngữ Đầu ra:** Đảm bảo ngôn ngữ đầu ra là {language}. Danh từ riêng (ví dụ: tên cá nhân, tên địa điểm, tên tổ chức) phải được giữ nguyên bằng ngôn ngữ gốc của chúng và không được dịch.

<Output>
"""

PROMPTS_VI["entity_continue_extraction_user_prompt"] = """---Nhiệm vụ---
Dựa trên nhiệm vụ trích xuất cuối cùng, xác định và trích xuất bất kỳ thực thể và mối quan hệ nào **bị bỏ sót hoặc định dạng không chính xác** từ văn bản đầu vào.

---Hướng dẫn---
1.  **Tuân thủ Nghiêm ngặt Định dạng Hệ thống:** Tuân thủ nghiêm ngặt tất cả các yêu cầu định dạng cho danh sách thực thể và mối quan hệ, bao gồm thứ tự đầu ra, dấu phân tách trường và xử lý danh từ riêng, như đã chỉ định trong hướng dẫn hệ thống.
2.  **Tập trung vào Sửa chữa/Bổ sung:**
    *   **KHÔNG** xuất ra lại các thực thể và mối quan hệ đã được trích xuất **chính xác và đầy đủ** trong nhiệm vụ cuối cùng.
    *   Nếu một thực thể hoặc mối quan hệ **bị bỏ sót** trong nhiệm vụ cuối cùng, hãy trích xuất và xuất ra ngay bây giờ theo định dạng hệ thống.
    *   Nếu một thực thể hoặc mối quan hệ **bị cắt ngắn, thiếu trường hoặc định dạng không chính xác** trong nhiệm vụ cuối cùng, hãy xuất ra lại phiên bản *đã sửa và hoàn chỉnh* theo định dạng đã chỉ định.
3.  **Định dạng Đầu ra - Thực thể:** Xuất ra tổng cộng 4 trường cho mỗi thực thể, được phân tách bằng `{tuple_delimiter}`, trên một dòng. Trường đầu tiên *phải* là chuỗi ký tự `entity`.
4.  **Định dạng Đầu ra - Mối quan hệ:** Xuất ra tổng cộng 5 trường cho mỗi mối quan hệ, được phân tách bằng `{tuple_delimiter}`, trên một dòng. Trường đầu tiên *phải* là chuỗi ký tự `relation`.
5.  **Chỉ Xuất ra Nội dung:** Chỉ xuất ra *danh sách* các thực thể và mối quan hệ đã trích xuất. Không bao gồm bất kỳ nhận xét mở đầu hoặc kết thúc, giải thích, hoặc văn bản bổ sung nào trước hoặc sau danh sách.
6.  **Tín hiệu Hoàn thành:** Xuất ra `{completion_delimiter}` là dòng cuối cùng sau khi tất cả các thực thể và mối quan hệ bị bỏ sót hoặc đã sửa có liên quan đã được trích xuất và trình bày.
7.  **Ngôn ngữ Đầu ra:** Đảm bảo ngôn ngữ đầu ra là {language}. Danh từ riêng (ví dụ: tên cá nhân, tên địa điểm, tên tổ chức) phải được giữ nguyên bằng ngôn ngữ gốc của chúng và không được dịch.

<Output>
"""

PROMPTS_VI["summarize_entity_descriptions"] = """---Vai trò---
Bạn là một Chuyên gia Đồ thị Tri thức, thành thạo trong việc tổ chức và tổng hợp dữ liệu.

---Nhiệm vụ---
Nhiệm vụ của bạn là tổng hợp một danh sách các mô tả của một thực thể hoặc mối quan hệ đã cho thành một bản tóm tắt duy nhất, toàn diện và mạch lạc.

---Hướng dẫn---
1. Định dạng Đầu vào: Danh sách mô tả được cung cấp ở định dạng JSON. Mỗi đối tượng JSON (đại diện cho một mô tả duy nhất) xuất hiện trên một dòng mới trong phần `Danh sách Mô tả`.
2. Định dạng Đầu ra: Mô tả đã được hợp nhất sẽ được trả về dưới dạng văn bản thuần túy, được trình bày trong nhiều đoạn văn, không có bất kỳ định dạng bổ sung hoặc nhận xét không liên quan nào trước hoặc sau bản tóm tắt.
3. Tính Toàn diện: Bản tóm tắt phải tích hợp tất cả thông tin chính từ *mọi* mô tả được cung cấp. Không bỏ qua bất kỳ sự kiện hoặc chi tiết quan trọng nào.
4. Ngữ cảnh: Đảm bảo bản tóm tắt được viết từ góc nhìn khách quan, ngôi thứ ba; nêu rõ tên của thực thể hoặc mối quan hệ để có sự rõ ràng và ngữ cảnh đầy đủ.
5. Ngữ cảnh & Tính khách quan:
  - Viết bản tóm tắt từ góc nhìn khách quan, ngôi thứ ba.
  - Nêu rõ tên đầy đủ của thực thể hoặc mối quan hệ ở đầu bản tóm tắt để đảm bảo tính rõ ràng và ngữ cảnh ngay lập tức.
  - Đối với tiếng Việt: Sử dụng cách diễn đạt tự nhiên, tránh dùng đại từ nhân xưng khi không cần thiết.
6. Xử lý Mâu thuẫn:
  - Trong trường hợp có mô tả mâu thuẫn hoặc không nhất quán, trước tiên hãy xác định xem những mâu thuẫn này có phát sinh từ nhiều thực thể hoặc mối quan hệ khác biệt, riêng biệt có cùng tên hay không.
  - Nếu xác định được các thực thể/mối quan hệ riêng biệt, hãy tóm tắt từng cái *riêng biệt* trong đầu ra tổng thể.
  - Nếu có mâu thuẫn trong một thực thể/mối quan hệ duy nhất (ví dụ: sự khác biệt lịch sử) tồn tại, hãy cố gắng hòa giải chúng hoặc trình bày cả hai quan điểm với sự không chắc chắn được ghi nhận.
7. Ràng buộc Độ dài: Tổng độ dài của bản tóm tắt không được vượt quá {summary_length} token, trong khi vẫn duy trì chiều sâu và tính đầy đủ.
8. Ngôn ngữ: Toàn bộ đầu ra phải được viết bằng {language}. Danh từ riêng (ví dụ: tên cá nhân, tên địa điểm, tên tổ chức) có thể được giữ nguyên bằng ngôn ngữ gốc nếu không có bản dịch phù hợp hoặc sẽ gây ra sự mơ hồ.
  - Toàn bộ đầu ra phải được viết bằng {language}.
  - Danh từ riêng (ví dụ: tên cá nhân, tên địa điểm, tên tổ chức) nên được giữ nguyên bằng ngôn ngữ gốc của chúng nếu bản dịch phù hợp, được chấp nhận rộng rãi không có sẵn hoặc sẽ gây ra sự mơ hồ.
  - Đối với tiếng Việt: Giữ nguyên tất cả dấu thanh và viết đúng chính tả.

---Đầu vào---
{description_type} Tên: {description_name}

Danh sách Mô tả:

```
{description_list}
```

---Đầu ra---
"""

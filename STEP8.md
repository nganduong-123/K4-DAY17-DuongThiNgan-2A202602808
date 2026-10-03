# Phân tích kết quả benchmark

## 1. Vì sao Advanced nhớ tốt hơn Baseline?

Ở Standard Benchmark, Baseline có recall `0.000`, còn Advanced đạt `1.000`. Baseline chỉ giữ lịch sử trong từng `thread_id`, nên khi câu hỏi được hỏi ở thread mới thì thông tin cũ không còn. Advanced dùng `extract_profile_updates()` để lấy fact ổn định, ghi vào `User.md`, rồi `_offline_response()` đọc lại file này ở thread mới. Vì vậy agent vẫn nhớ đúng tên, nơi ở, nghề nghiệp và sở thích. Khi người dùng đính chính từ Đà Nẵng sang Huế hoặc từ backend sang MLOps, giá trị mới thay thế giá trị cũ.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Trong bộ Standard, Advanced dùng `3.028` agent token và xử lý `24.730` prompt token, cao hơn Baseline là `1.289` và `13.591`. Lý do là mỗi lượt Advanced còn đọc profile, trích xuất fact và mang thêm memory vào context. Chi phí này có ích cho recall dài hạn, nhưng với hội thoại ngắn thì phần memory bổ sung có thể chưa tạo ra lợi ích đủ lớn.

## 3. Compact giúp gì khi context dài?

Ở stress test, prompt token của Baseline là `21.944`, còn Advanced là `12.757`, giảm `9.187` token, khoảng `41,9%`. Advanced đã compact `3` lần: nội dung cũ được chuyển thành summary và chỉ giữ các message gần nhất. Vì vậy lợi ích chính nằm ở lượng prompt phải xử lý, không phải lượng token agent sinh ra. Thực tế agent token của Advanced vẫn cao hơn (`565` so với `217`) vì câu trả lời của nó chứa thông tin recall đầy đủ hơn.

## 4. Memory file tăng ra sao và có rủi ro gì?

File memory tăng `264 bytes` ở bộ Standard và `208 bytes` ở stress test. Structured profile làm file tăng chậm vì mỗi field chỉ giữ một giá trị mới nhất. Tuy nhiên nếu số field hoặc số người dùng tăng, dung lượng vẫn có thể phình ra. Rủi ro lớn hơn là lưu nhầm câu hỏi, câu đùa hoặc fact đã cũ; ngoài ra profile còn có thể chứa dữ liệu cá nhân nên cần giới hạn quyền truy cập và thời gian lưu.

## Bonus: structured extraction và conflict handling

Em dùng các field cố định như `name`, `location`, `profession`, `response_style` và chỉ ghi khi câu có dấu hiệu đủ chắc chắn. Câu hỏi recall và các chi tiết nhiễu như đi họp ở Hà Nội hoặc đùa làm product manager không được lưu thành fact. Nếu cùng một field có correction, `upsert_fact()` thay giá trị cũ thay vì giữ cả hai. Cách này giúp recall đạt `1.000` và giữ file gọn, nhưng rule dựa trên mẫu câu có thể bỏ sót cách diễn đạt mới; hệ thống thật nên bổ sung confidence score và bước xác nhận với người dùng.

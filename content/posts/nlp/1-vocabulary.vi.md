---
author: "Le Van Dong"
title: "Vocabulary in NLP"
date: "2025-08-20"
tags: ["AI", "Machine Learning"]
categories: ["Natural Language Processing"]
---

Ngôn ngữ tự nhiên của con người tồn tại ở hai dạng chính: giọng nói và văn bản. Giọng nói là nền tảng cho các ứng dụng như trợ lý ảo Siri hay Google Assistant. Trong khi đó, dữ liệu văn bản là đầu vào cốt lõi cho phần lớn các bài toán Xử lý Ngôn ngữ Tự nhiên (NLP).

Vấn đề cơ bản của NLP là máy tính không thể trực tiếp hiểu được ngôn ngữ của con người. Do đó, chúng ta phải chuyển đổi dữ liệu văn bản từ dạng chuỗi ký tự sang một định dạng số học mà máy tính có thể xử lý được. Quá trình này là bước tiền xử lý thiết yếu để làm đầu vào cho các mô hình học máy.

## 1\. Corpus và Vocabulary

- **Text Corpus:** Là một tập hợp lớn các văn bản được sử dụng cho một mục đích cụ thể. Ví dụ, nếu chúng ta muốn xây dựng một mô hình phân loại tin tức, thì tập hợp tất cả các bài báo dùng để huấn luyện (train) và đánh giá (evaluate) mô hình chính là text corpus.

- **Vocabulary (Bộ từ vựng):** Là tập hợp tất cả các từ *duy nhất* (unique) xuất hiện trong text corpus. Việc xây dựng và sử dụng bộ từ vựng này là một bước cốt lõi trong hầu hết các quy trình NLP.

Ví dụ, với corpus là câu: "Tôi yêu AI. AI rất tuyệt.", bộ từ vựng sẽ là: `{"Tôi", "yêu", "AI", "rất", "tuyệt"}`.

## 2\. Tokenization

Trước khi xây dựng bộ từ vựng, văn bản thô cần được phân tách thành các đơn vị cơ bản hơn.

- **Token:** Là đơn vị nhỏ nhất của văn bản mà mô hình sẽ xử lý. Tùy thuộc vào phương pháp được chọn, một token có thể là một từ, một ký tự, hoặc một tiền tố/hậu tố (subword).

- **Tokenization:** Là quá trình phân tách một chuỗi văn bản thành một danh sách các token. Đây là bước đầu tiên và quan trọng trong việc chuẩn bị dữ liệu.

## 3\. Xây dựng Từ điển (Vocabulary)

Sau khi đã có bộ từ vựng từ corpus, bước tiếp theo là xây dựng một "từ điển" (dictionary) để ánh xạ mỗi token sang một giá trị số duy nhất (thường là một số nguyên ID). Quá trình này tương tự như việc con người học từ mới và ghi nhớ từ vựng; mô hình học máy cũng chỉ có thể "hiểu" được những token đã có trong từ điển của nó. Những từ chưa từng xuất hiện sẽ được xem là "out-of-vocabulary" (OOV) hay từ không xác định.

Quy trình tạo từ điển thường tuân theo các quy tắc sau:

1. **Thống kê tần suất:** Đếm số lần xuất hiện của mỗi token trong toàn bộ corpus.
2. **Sắp xếp:** Sắp xếp các token theo thứ tự tần suất xuất hiện giảm dần. Nếu hai token có cùng tần suất, chúng sẽ được sắp xếp theo thứ tự bảng chữ cái.
3. **Gán ID:** Gán một ID số nguyên tuần tự cho mỗi token theo thứ tự đã sắp xếp. Thông thường, ID `0` được dành riêng cho việc đệm chuỗi (padding) hoặc cho các token không xác định (OOV).

### Ví dụ minh họa

Giả sử chúng ta có một text corpus gồm hai câu:
`["bob ăn táo và bưởi", "fred ăn táo!"]`

**Bước 1: Tiền xử lý và Tokenization**

- Chuyển thành chữ thường và loại bỏ dấu câu.
- Corpus sau khi xử lý: `["bob", "ăn", "táo", "và", "bưởi", "fred", "ăn", "táo"]`

**Bước 2: Xây dựng bộ từ vựng và đếm tần suất**

- Bộ từ vựng (vocabulary): `{"bob", "ăn", "táo", "và","bưởi", "fred"}`
- Tần suất xuất hiện:
  - ăn: 2
  - táo: 2
  - bob: 1
  - bưởi: 1
  - fred: 1
  - và: 1

**Bước 3: Sắp xếp và tạo từ điển**

- Sắp xếp theo tần suất giảm dần. Vì "ăn" và "táo" có cùng tần suất, ta xếp chúng theo thứ tự bảng chữ cái. Tương tự với các từ còn lại.
- Thứ tự ưu tiên: `(ăn: 2), (táo: 2), (bob: 1), (bưởi: 1), (fred: 1), (và: 1)`
- Gán ID (bắt đầu từ 1, dành 0 cho token OOV):

```json
{
  "ăn": 1,
  "táo": 2,
  "bob": 3,
  "bưởi": 4,
  "fred": 5,
  "và": 6
}
```

*Lưu ý: Thứ tự của các từ cùng tần suất có thể thay đổi tùy thuộc vào cách triển khai, nhưng quy tắc chung là nhất quán.*

**Bước 4: Mã hóa văn bản mới**
Bây giờ, khi có một câu mới, chúng ta có thể dùng từ điển trên để chuyển nó thành một chuỗi các con số:

- Văn bản mới: `["bob ăn bưởi", "fred cũng ăn bưởi"]`
- Từ "cũng" không có trong từ điển, nên nó sẽ được xem là token OOV và nhận ID là `0`.
- Biểu diễn dạng số: `[[3, 1, 4], [5, 0, 1, 4]]`

Vậy là chúng ta đã chuyển đổi thành công dữ liệu văn bản thành định dạng số, sẵn sàng để làm đầu vào cho các mô hình NLP.

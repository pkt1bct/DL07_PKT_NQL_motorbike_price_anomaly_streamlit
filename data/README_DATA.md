# Dữ liệu dùng cho project

## 1. Dữ liệu chính cần bổ sung

Đặt file dữ liệu huấn luyện/phân tích của P2 tại:

```text
data/data_motobikes.xlsx
```

File này là dữ liệu tin đăng xe máy cũ trên Chợ Tốt, được dùng cho phần Business Problem, Evaluation & Report và các biểu đồ phân tích.

## 2. File kiểm tra hàng loạt

Project đã tạo sẵn:

```text
data/sample_batch_upload.csv
```

Các cột bắt buộc:

- `thuong_hieu`
- `dong_xe`
- `loai_xe`
- `nam_dang_ky`
- `gia_rao`

Các cột tùy chọn:

- `so_km`
- `dung_tich_cc`
- `xuat_xu`
- `quan`
- `tieu_de`
- `mo_ta`

## 3. Lưu ý đơn vị

- `gia_rao`: đồng Việt Nam, ví dụ `28000000`.
- `so_km`: km.
- `dung_tich_cc`: cc.

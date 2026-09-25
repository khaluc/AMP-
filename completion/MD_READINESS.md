# Tình trạng mô phỏng phân tử

Chưa chạy MARTINI/MD. Việc chọn 12 đại diện hình học trong Phase 3 đã hoàn thành, nhưng chưa có hệ mô phỏng
được xác định cho từng đại diện. Windows không tìm thấy `gmx`/`gmx_mpi`; danh sách WSL chỉ có `docker-desktop`.
Không cài thêm hệ điều hành hay thay đổi cấu hình Docker hiện tại chỉ để tạo một kết quả mô phỏng chưa có cơ sở.

Những đầu vào còn thiếu trước khi có thể chạy và diễn giải mô phỏng:

1. Cấu trúc khởi đầu và cách xác nhận cấu trúc cho từng peptide; trạng thái đầu mút và protonation tương ứng.
2. Phiên bản force field, mapping, topology và tham số tương thích đã kiểm tra.
3. Hệ màng, thành phần, điều kiện mô phỏng và đối chứng phù hợp với câu hỏi nghiên cứu.
4. Giao thức cân bằng, số lần lặp, tiêu chí hội tụ và tiêu chí phân tích được xác định trước.

Các mục này không thể suy ra chỉ từ cluster ID hoặc chuỗi FASTA. File `artifacts/md_readiness.json` liệt kê trạng thái
của từng đại diện để theo dõi phần còn thiếu. File đó không phải cấu hình chạy và không chứa số liệu mô phỏng giả.
Một quỹ đạo ngắn hoặc một cụm embedding không đủ để gán chắc chắn cơ chế cho toàn bộ cụm.

Nguồn kỹ thuật: [GROMACS topology documentation](https://manual.gromacs.org/current/reference-manual/topologies/topologies.html).

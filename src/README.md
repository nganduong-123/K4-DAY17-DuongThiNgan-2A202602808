# Source implementation

Thư mục này chứa bản triển khai hoàn chỉnh cho Day 17:

- `model_provider.py`: chuẩn hóa và khởi tạo 6 provider ở chế độ live
- `config.py`: cấu hình đường dẫn, model và compact memory
- `memory_store.py`: `User.md`, structured extraction, conflict handling và compact memory
- `agent_baseline.py`: chỉ nhớ trong một thread
- `agent_advanced.py`: persistent profile và compact memory
- `benchmark.py`: standard benchmark và long-context stress benchmark
- `test_agents.py`: 4 test hành vi bắt buộc

Hai agent mặc định được benchmark ở chế độ offline để kết quả ổn định và không phụ thuộc API key.

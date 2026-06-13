import pandas as pd

# =====================
# 1. CPU
# =====================
cpu = pd.read_csv("data/cpu_usage.csv")

cpu = cpu.rename(columns={
    "Time": "timestamp",
    "系统使用率": "cpu_system",
    "磁盘IO使用率": "cpu_iowait",
    "总使用率": "cpu_total"
})

cpu = cpu[
    [
        "timestamp",
        "cpu_system",
        "cpu_iowait",
        "cpu_total"
    ]
]

# =====================
# 2. Memory
# =====================
memory = pd.read_csv("data/memory_usage.csv")

memory = memory.rename(columns={
    "Time": "timestamp",
    "192.168.49.2:9100_已用(total-cache-buf-free)": "memory_used"
})

memory = memory[
    [
        "timestamp",
        "memory_used"
    ]
]

# =====================
# 3. System Load
# =====================
load = pd.read_csv("data/system_load.csv")

load = load.rename(columns={
    "Time": "timestamp",
    "1分钟负载": "load_1m",
    "5分钟负载": "load_5m"
})

load = load[
    [
        "timestamp",
        "load_1m",
        "load_5m"
    ]
]

# =====================
# 4. Network
# =====================
network = pd.read_csv("data/network_bandwidth.csv")

network = network.rename(columns={
    "Time": "timestamp",
    "eth0_in下载": "network_in",
    "eth0_out上传": "network_out"
})

network = network[
    [
        "timestamp",
        "network_in",
        "network_out"
    ]
]

# =====================
# 5. Disk Bytes
# =====================
disk = pd.read_csv("data/disk_io_bytes.csv")

disk["disk_write_bytes"] = (
    disk["sda_写入"] +
    disk["sdb_写入"] +
    disk["sdc_写入"] +
    disk["sdd_写入"] +
    disk["sde_写入"] +
    disk["sdf_写入"]
)

disk = disk.rename(columns={
    "Time": "timestamp"
})

disk = disk[
    [
        "timestamp",
        "disk_write_bytes"
    ]
]

# =====================
# 6. IO Latency
# =====================
latency = pd.read_csv("data/io_latency.csv")

write_cols = [
    "sda_写入",
    "sdb_写入",
    "sdc_写入",
    "sdd_写入",
    "sde_写入",
    "sdf_写入"
]

latency["io_latency"] = latency[write_cols].fillna(0).sum(axis=1)

latency = latency.rename(columns={
    "Time": "timestamp"
})

latency = latency[
    [
        "timestamp",
        "io_latency"
    ]
]

# =====================
# 7. IO Wait Ratio
# =====================
wait_ratio = pd.read_csv("data/io_wait_ratio.csv")

io_cols = [
    "sda_每秒I/O操作%",
    "sdb_每秒I/O操作%",
    "sdc_每秒I/O操作%",
    "sdd_每秒I/O操作%",
    "sde_每秒I/O操作%",
    "sdf_每秒I/O操作%"
]

wait_ratio["io_wait_ratio"] = wait_ratio[io_cols].sum(axis=1)

wait_ratio = wait_ratio.rename(columns={
    "Time": "timestamp"
})

wait_ratio = wait_ratio[
    [
        "timestamp",
        "io_wait_ratio"
    ]
]

# =====================
# 8. Context Switch
# =====================
context = pd.read_csv("data/context_switch_per_sec.csv")

context = context.rename(columns={
    "Time": "timestamp",
    "每秒上下文切换次数": "context_switch"
})

context = context[
    [
        "timestamp",
        "context_switch"
    ]
]

# =====================
# 9. Socket
# =====================
socket = pd.read_csv("data/socket_connections.csv")

socket = socket.rename(columns={
    "Time": "timestamp",
    "CurrEstab": "tcp_estab"
})

socket = socket[
    [
        "timestamp",
        "tcp_estab"
    ]
]

# =====================
# Merge
# =====================

dfs = [
    cpu,
    memory,
    load,
    network,
    disk,
    latency,
    wait_ratio,
    context,
    socket
]

dataset = dfs[0]

for df in dfs[1:]:
    dataset = pd.merge(
        dataset,
        df,
        on="timestamp",
        how="outer"
    )

# 只删除所有字段都为空的行
dataset = dataset.dropna(how="all")

# 时间排序
dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])

dataset = dataset.sort_values(
    by="timestamp"
)

dataset.to_csv(
    "dataset.csv",
    index=False
)

print(dataset.head())
print()
print("Shape:", dataset.shape)
print("dataset.csv saved")
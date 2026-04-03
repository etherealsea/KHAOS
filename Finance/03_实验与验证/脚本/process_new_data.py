import sys, os, glob
sys.path.append(os.path.join(os.getcwd(), 'Finance', '02_核心代码', '源代码'))
from khaos.数据处理.data_processor import process_multi_timeframe

def main():
    train_data_dir = r"d:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed\training_ready"
    os.makedirs(train_data_dir, exist_ok=True)
    
    files_to_process = [
        r"d:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed\Crypto\BTCUSD_5m.csv",
        r"d:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed\Crypto\ETHUSD_5m.csv",
        r"d:\《基于物理信息增强神经网络（PI-KAN）的金融时间序列相变探测研究》\Finance\01_数据中心\03_研究数据\research_processed\Index\ESUSD_5m.csv"
    ]
    
    for f in files_to_process:
        if os.path.exists(f):
            print(f"Processing {f}...")
            new_files = process_multi_timeframe(f, train_data_dir)
            print(f"  Generated: {new_files}")
        else:
            print(f"Not found: {f}")

if __name__ == "__main__":
    main()

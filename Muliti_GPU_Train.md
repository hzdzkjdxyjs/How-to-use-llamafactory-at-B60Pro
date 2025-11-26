# 怎么用xpu的后端通信库进行通信
 - DeepSpeed在底层上来说是进程之间进行通信，说起来也很搞笑啊，我在测试DeepSpeed进行多卡训练的时候，根本没装任何和通信有关的库
 - 所以现在我们希望训练的时候能从底层调用xpu的优化后的通信库来训练

---

## 验证您的系统是否可以支持多卡通信
 - gloo通信，（创建文件 执行脚本）

```bash
import torch.distributed as dist

def main():
    dist.init_process_group("gloo")
    print("Backend =", dist.get_backend())
    print("Rank =", dist.get_rank())

if __name__ == "__main__":
    main()
````
<img width="1241" height="406" alt="image" src="https://github.com/user-attachments/assets/5b234187-4ba6-49cb-ab4d-afd97ea14a22" />

 - oneccl通信，（创建文件 执行脚本）

```bash
import oneccl_bindings_for_pytorch 
import torch.distributed as dist

def main():
    dist.init_process_group("ccl")
    print("Backend =", dist.get_backend(), "Rank =", dist.get_rank())

if __name__ == "__main__":
    main()
````

<img width="1241" height="406" alt="image" src="https://github.com/user-attachments/assets/ff3a9ce4-6c9c-4f5a-bf2b-f17deea172e7" />



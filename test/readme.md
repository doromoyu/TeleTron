# TeleTron Testing

## 测试分布式方法
* 在teletron中很多方法需要在分布式场景下才能充分测试，比如split, gather, all-to-all等，因此需要测试用例本身包含多进程。然而pytest本身对多进程测试用例的支持度不好，1）子进程没法使用self.assertTrue这样的断言来告知pytest测试用例的成败 2）子进程中的报错pytest不会捕捉，所以当子进程报错时从pytest层面只能看到进程自己退出了，不知道是什么报错，不利于分析和调试。因此在teletron testing中开发了一个简单的`spawn`接口便于启动pytorch分布式进程并且便于和pytest框架做交互。

接口描述：
```
spawn(nprocs: int, func: Callable, *args) -> multiprocessing.Queue
```
入参中nprocs是进程数，func是每个进程要运行的程序，args是他们的参数，spawn接口会给func传入rank和world_size作为第一个和第二个位置参数，并且传入一个multiprocessing.Queue的对象q作为第三个参数，args（如有）会作为第四个及以后的参数。

multiprocessing.Queue作为子进程和父进程的桥梁，在func中可以通过q.put()接口向父进程传递字符串信息，然后在父进程中通过q.get()来接受所有子进程的信息，并且以此判定测试用例的成败。unit_test/core/test_parallel_state.py中有一个通过此机制实现子进程测试的极简示例，看一下你应该就明白了。另外，在实际测试中，你一般需要初始化pytorch的多进程（使用init_process_group）并且初始化megatron进程组（使用initialize_model_parallel），你可以使用传入的rank和world_size参数来做到这一点。unit_test/core/context_parallel/test_context_parallel_model.py中有相关示例可以参考。

附注：在pytest中直接print信息会被吞掉，要用logging.info(str)来打印调试信息，并且运行pytest时要加-o log_cli=true -o log_cli_level=INFO参数以设置pytest将打印信息输出到屏幕上。
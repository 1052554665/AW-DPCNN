## 1. IOP 不喜欢 abstract/keywords

 `.bib` 里大量：

```bibtex
abstract =
keywords =
```

实际上：

- `iopart-num` 不使用
    
- 会让 `.bib` 极度臃肿
    
- 增加编译负担
    
- Git diff 很乱
    

删除。


## 2. month 字段格式不统一

例如：

```bibtex
date = {2024-07}
```

和：

```bibtex
year = {2024}
```

混用。

修改为BibTeX 最稳定的：

```bibtex
year = {2024},
month = jul,
```

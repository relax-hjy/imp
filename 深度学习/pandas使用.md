创建DataFrame ：

```python
# 通过类创建对象
x=pd.DataFrame()

y=pd.DataFrame("a":[1,2,3,4],"b":[5,6,7,8])

# 通过文件导入函数创建：

o=pd.read_csv("data.csv")

p=pd.read_excel("data.xlsx",sheet_name="")
```





```
df.count()           # 统计总数
df.isnull().sum()    # 统计空值个数，is_null()返回布尔series
df.dropna().count()

df[df["A"].isnull()]    # 拿到列的空值
df.dropna(subset='['A'])  # 拿到列的非空值
```


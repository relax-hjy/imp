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
df.count()           # 统计总行数
df.isnull().sum()    # 统计有空值的行数，is_null()返回布尔series
df.dropna().count()  # 统计非空值的行数

df[df["A"].isnull()]    # 拿到某一字段为空的所有行
df.dropna(subset='['A'])  # 拿到某一字段不为空的所有行
```


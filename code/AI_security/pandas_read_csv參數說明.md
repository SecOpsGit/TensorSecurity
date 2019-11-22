#
```
Pandas操作資料的函數
https://ithelp.ithome.com.tw/articles/10193421
```

### pandas.read_csv()
```
pandas.read_csv(
filepath_or_buffer: Union[str, pathlib.Path, IO[~AnyStr]], 
sep=',', 
delimiter=None, 
header='infer', 
names=None, 
index_col=None, 
usecols=None, 
squeeze=False, 
prefix=None, 
mangle_dupe_cols=True, 
dtype=None, 
engine=None, 
converters=None, 
true_values=None, 
false_values=None, 
skipinitialspace=False, 
skiprows=None, 
skipfooter=0, 
nrows=None, 
na_values=None, 
keep_default_na=True, 
na_filter=True, 
verbose=False, 
skip_blank_lines=True, 
parse_dates=False, 
infer_datetime_format=False, 
keep_date_col=False, 
date_parser=None, 
dayfirst=False, 
cache_dates=True, 
iterator=False, 
chunksize=None, 
compression='infer', 
thousands=None, 
decimal=b'.', 
lineterminator=None, 
quotechar='"', 
quoting=0, 
doublequote=True, 
escapechar=None, 
comment=None, 
encoding=None, 
dialect=None, 
error_bad_lines=True, 
warn_bad_lines=True, 
delim_whitespace=False, 
low_memory=True, 
memory_map=False, 
float_precision=None)

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
https://www.cnblogs.com/datablog/p/6127000.html
```

```

filepath_or_buffer : str，pathlib。str, pathlib.Path, py._path.local.LocalPath or any object with a read() method (such as a file handle or StringIO)
可以是URL，可用URL類型包括：http, ftp, s3和文件。對於多文件正在準備中
本地檔讀取實例：://localhost/path/to/table.csv
 
sep : str, default ‘,’
指定分隔符號。如果不指定參數，則會嘗試使用逗號分隔。
分隔符號長於一個字元並且不是‘\s+’,將使用python的語法分析器。並且忽略資料中的逗號。規則運算式例子：'\r\t'
 
delimiter : str, default None
定界符，備選分隔符號（如果指定該參數，則sep參數失效）
 
delim_whitespace : boolean, default False.
指定空格(例如’ ‘或者’ ‘)是否作為分隔符號使用，等效於設定sep='\s+'。
如果這個參數設定為Ture那麼delimiter 參數失效。
在新版本0.18.1支援
 
header : int or list of ints, default ‘infer’
指定行數用來作為列名，資料開始行數。如果檔中沒有列名，則默認為0，否則設置為None。
如果明確設定header=0 就會替換掉原來存在列名。header參數可以是一個list例如：[0,1,3]，
這個list表示將檔中的這些行作為列標題（意味著每一列有多個標題），介於中間的行將被忽略掉
（例如本例中的2；本例中的資料1,2,4行將被作為多級標題出現，第3行資料將被丟棄，dataframe的資料從第5行開始。）。
注意：如果skip_blank_lines=True 那麼header參數忽略注釋行和空行，所以header=0表示第一行資料而不是檔的第一行。
 
names : array-like, default None
用於結果的列名列表，如果資料檔案中沒有列標題行，就需要執行header=None。
默認列表中不能出現重複，除非設定參數mangle_dupe_cols=True。
 
index_col : int or sequence or False, default None
用作行索引的列編號或者列名，如果給定一個序列則有多個行索引。
如果檔不規則，行尾有分隔符號，則可以設定index_col=False 來是的pandas不適用第一列作為行索引。
 
usecols : array-like, default None
返回一個資料子集，該清單中的值必須可以對應到檔中的位置（數位可以對應到指定的列）
或者是字元傳為檔中的列名。例如：usecols有效參數可能是 [0,1,2]或者是 [‘foo’, ‘bar’, ‘baz’]。
使用這個參數可以加快載入速度並降低記憶體消耗。
 
as_recarray : boolean, default False
不贊成使用：該參數會在未來版本移除。請使用pd.read_csv(...).to_records()替代。
返回一個Numpy的recarray來替代DataFrame。如果該參數設定為True。
將會優先squeeze參數使用。並且行索引將不再可用，索引列也將被忽略。
 
squeeze : boolean, default False
如果檔值包含一列，則返回一個Series
 
prefix : str, default None
在沒有列標題時，給列添加首碼。例如：添加‘X’ 成為 X0, X1, ...
 
mangle_dupe_cols : boolean, default True
重複的列，將‘X’...’X’表示為‘X.0’...’X.N’。如果設定為false則會將所有重名列覆蓋。
 
dtype : Type name or dict of column -> type, default None
每列資料的資料類型。例如 {‘a’: np.float64, ‘b’: np.int32}
 
engine : {‘c’, ‘python’}, optional
Parser engine to use. The C engine is faster while 
the python engine is currently more feature-complete.
使用的分析引擎。可以選擇C或者是python。C引擎快但是Python引擎功能更加完備。
 
converters : dict, default None
列轉換函數的字典。key可以是列名或者列的序號。
 
true_values : list, default None
Values to consider as True
 
false_values : list, default None
Values to consider as False
 
skipinitialspace : boolean, default False
忽略分隔符號後的空白（默認為False，即不忽略）.
 
skiprows : list-like or integer, default None
需要忽略的行數（從檔開始處算起），或需要跳過的行號列表（從0開始）。
 
skipfooter : int, default 0
從檔案結尾部開始忽略。 (c引擎不支援)
 
skip_footer : int, default 0
不推薦使用：建議使用skipfooter ，功能一樣。
 
nrows : int, default None
需要讀取的行數（從檔頭開始算起）。
 
na_values : scalar, str, list-like, or dict, default None
一組用於替換NA/NaN的值。如果傳參，需要制定特定列的空值。
默認為‘1.#IND’, ‘1.#QNAN’, ‘N/A’, ‘NA’, ‘NULL’, ‘NaN’, ‘nan’`.
 
keep_default_na : bool, default True
如果指定na_values參數，並且keep_default_na=False，那麼默認的NaN將被覆蓋，否則添加。
 
na_filter : boolean, default True
是否檢查丟失值（空字串或者是空值）。對於大檔來說資料集中沒有空值，
設定na_filter=False可以提升讀取速度。
 
verbose : boolean, default False
是否列印各種解析器的輸出資訊，例如：“非數值列中缺失值的數量”等。
 
skip_blank_lines : boolean, default True
如果為True，則跳過空行；否則記為NaN。
 
parse_dates : boolean or list of ints or names or list of lists or dict, default False
boolean. True -> 解析索引
list of ints or names. e.g. If [1, 2, 3] -> 解析1,2,3列的值作為獨立的日期列；
list of lists. e.g. If [[1, 3]] -> 合併1,3列作為一個日期列使用
dict, e.g. {‘foo’ : [1, 3]} -> 將1,3列合併，並給合併後的列起名為"foo"
 
infer_datetime_format : boolean, default False
如果設定為True並且parse_dates 可用，那麼pandas將嘗試轉換為日期類型，
如果可以轉換，轉換方法並解析。在某些情況下會快5~10倍。
 
keep_date_col : boolean, default False
如果連接多列解析日期，則保持參與連接的列。默認為False。
 
date_parser : function, default None
用於解析日期的函數，預設使用dateutil.parser.parser來做轉換。
Pandas嘗試使用三種不同的方式解析，如果遇到問題則使用下一種方式。
1.使用一個或者多個arrays（由parse_dates指定）作為參數；
2.連接指定多列字串作為一個列作為參數；
3.每行調用一次date_parser函數來解析一個或者多個字串（由parse_dates指定）作為參數。
 
dayfirst : boolean, default False
DD/MM格式的日期類型
 
iterator : boolean, default False
返回一個TextFileReader 物件，以便逐塊處理檔。
 
chunksize : int, default None
文件塊的大小， See IO Tools docs for more informationon iterator and chunksize.
 
compression : {‘infer’, ‘gzip’, ‘bz2’, ‘zip’, ‘xz’, None}, default ‘infer’
直接使用磁片上的壓縮檔。
如果使用infer參數，則使用 gzip, bz2, zip或者解壓檔案名中以‘.gz’, ‘.bz2’, ‘.zip’, or ‘xz’這些為尾碼的檔，
否則不解壓。如果使用zip，那麼ZIP包中國必須只包含一個檔。設置為None則不解壓。
新版本0.18.1版本支援zip和xz解壓
 
thousands : str, default None
千分位分割符，如“，”或者“."
 
decimal : str, default ‘.’
字元中的小數點 (例如：歐洲資料使用’，‘).
 
float_precision : string, default None
Specifies which converter the C engine should use for floating-point values. 
The options are None for the ordinary converter, high for the high-precision converter, 
and round_trip for the round-trip converter.
指定
 
lineterminator : str (length 1), default None
行分割符，只在C解析器下使用。
 
quotechar : str (length 1), optional
引號，用作標識開始和解釋的字元，引號內的分割符將被忽略。
 
quoting : int or csv.QUOTE_* instance, default 0
控制csv中的引號常量。可選 QUOTE_MINIMAL (0), QUOTE_ALL (1), QUOTE_NONNUMERIC (2) or QUOTE_NONE (3)
 
doublequote : boolean, default True
雙引號，當單引號已經被定義，並且quoting 參數不是QUOTE_NONE的時候，
使用雙引號表示引號內的元素作為一個元素使用。
 
escapechar : str (length 1), default None
當quoting 為QUOTE_NONE時，指定一個字元使的不受分隔符號限值。
 
comment : str, default None
標識著多餘的行不被解析。如果該字元出現在行首，這一行將被全部忽略。
這個參數只能是一個字元，空行（就像skip_blank_lines=True）注釋行被header和skiprows忽略一樣。
例如如果指定comment='#' 解析‘#empty\na,b,c\n1,2,3’ 以header=0 那麼返回結果將是以’a,b,c'作為header。
 
encoding : str, default None
指定字元集類型，通常指定為'utf-8'. List of Python standard encodings
 
dialect : str or csv.Dialect instance, default None
如果沒有指定特定的語言，如果sep大於一個字元則忽略。具體查看csv.Dialect 文檔
 
tupleize_cols : boolean, default False
Leave a list of tuples on columns as is (default is to convert to a Multi Index on the columns)
 
error_bad_lines : boolean, default True
如果一行包含太多的列，那麼默認不會返回DataFrame ，
如果設置成false，那麼會將改行剔除（只能在C解析器下使用）。
 
warn_bad_lines : boolean, default True
如果error_bad_lines =False，並且warn_bad_lines =True 
那麼所有的“bad lines”將會被輸出（只能在C解析器下使用）。
 
low_memory : boolean, default True
分塊載入到記憶體，再低記憶體消耗中解析。但是可能出現類型混淆。
確保類型不被混淆需要設置為False。或者使用dtype 參數指定類型。
注意使用chunksize 或者iterator 參數分塊讀入會將整個檔讀入到一個Dataframe，而忽略類型（只能在C解析器中有效）
 
buffer_lines : int, default None
不推薦使用，這個參數將會在未來版本移除，因為他的值在解析器中不推薦使用
 
compact_ints : boolean, default False
不推薦使用，這個參數將會在未來版本移除
如果設置compact_ints=True ，那麼任何有整數類型構成的列將被按照最小的整數類型存儲，
是否有符號將取決於use_unsigned 參數
 
use_unsigned : boolean, default False
不推薦使用：這個參數將會在未來版本移除
如果整數列被壓縮(i.e. compact_ints=True)，指定被壓縮的列是有符號還是無符號的。

memory_map : boolean, default False
如果使用的檔在記憶體內，那麼直接map檔使用。使用這種方式可以避免檔再次進行IO操作。

```

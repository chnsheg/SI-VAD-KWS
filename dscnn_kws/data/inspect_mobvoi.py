from modelscope.msdatasets import MsDataset

ds = MsDataset.load("daydream-factory/mobvoi_hotword_dataset")

print("type(ds):", type(ds))

if isinstance(ds, dict):
    print("splits:", ds.keys())
    for split_name, split_ds in ds.items():
        print("\n==========", split_name, "==========")
        print("type:", type(split_ds))
        print("len:", len(split_ds))
        item = split_ds[0]
        print("sample type:", type(item))
        print("sample keys:", item.keys() if isinstance(item, dict) else "not dict")
        print("sample:", item)
else:
    print("len:", len(ds))
    item = ds[0]
    print("sample type:", type(item))
    print("sample keys:", item.keys() if isinstance(item, dict) else "not dict")
    print("sample:", item)
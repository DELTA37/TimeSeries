{
  "kind"            : "test",
  "train_path"      : "./data_train",
  "test_path"       : "./data_test",
  "result_test_dir" : "./results",
  "restore_path"    : "./restore",
  "restore_file"    : "checkpoint.ckpt.tar",
  "restore"         : false,
  "batch_size"      : 10,
  "summary_path"    : "./summary",
  "shuffle"         : true,
  "transforms"      : {
    "ToTensor" : true
  }
}

{
  "test_num_batches": "10",
  "opt"             : "SGD",
  "num_epoch"       : "4",
  "auto_save"       : "100",
  "learning_rate"   : 0.0001,
  "kind"            : "test",
  "train_path"      : "~/data/CatDog",
  "test_path"       : "~/data/CatDog",
  "result_test_dir" : "./results",
  "restore_path"    : "./restore",
  "restore_file"    : "checkpoint.ckpt.tar",
  "restore"         : false,
  "batch_size"      : 10,
  "summary_path"    : "./summary",
  "shuffle"         : true,
  "deprecate"       : ["batch_size"],
  "transforms"      : {
    "ToTensor" : true
  }
}

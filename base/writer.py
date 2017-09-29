import tensorflow as tf

# tensrflow backend for loss visualizing 

def write_summaries(stat_dict, path):
    if not hasattr(write_summaries, 'n'):
        write_summaries.n = dict()

    if path not in write_summaries.n.keys():
        write_summaries.n[path] = 0

    write_summaries.n[path] += 1

    if not hasattr(write_summaries, 'sess'):
        write_summaries.sess = dict()

    if not hasattr(write_summaries, 'vars'):
        write_summaries.vars = dict()

    if path not in write_summaries.sess.keys():
        write_summaries.sess[path] = tf.Session(), tf.summary.FileWriter(path)
    
    for name, val in stat_dict.items():
        if name not in write_summaries.vars.keys():
            write_summaries.vars[name] = tf.Variable(val, name=name)
            tf.summary.scalar(name, write_summaries.vars[name])


    summary_op = tf.summary.merge_all()
    sw = write_summaries.sess[path][1] 
    sess = write_summaries.sess[path][0]
    
    feed_dict = {}
    for name, val in stat_dict.items():
        feed_dict[write_summaries.vars[name]] = val
    
    s = sess.run([summary_op], feed_dict)[0]

    sw.add_summary(s, write_summaries.n[path])
    sw.flush()


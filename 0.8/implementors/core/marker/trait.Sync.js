(function() {var implementors = {};
implementors["ndarray"] = ["impl&lt;S, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> for <a class=\"struct\" href=\"ndarray/struct.ArrayBase.html\" title=\"struct ndarray::ArrayBase\">ArrayBase</a>&lt;S, D&gt; <span class=\"where fmt-newline\">where S: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> + <a class=\"trait\" href=\"ndarray/trait.Data.html\" title=\"trait ndarray::Data\">Data</a>, D: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a></span>","impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> for <a class=\"struct\" href=\"ndarray/struct.Iter.html\" title=\"struct ndarray::Iter\">Iter</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a>, D: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a></span>","impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> for <a class=\"struct\" href=\"ndarray/struct.IndexedIter.html\" title=\"struct ndarray::IndexedIter\">IndexedIter</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a>, D: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a></span>","impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> for <a class=\"struct\" href=\"ndarray/struct.InnerIter.html\" title=\"struct ndarray::InnerIter\">InnerIter</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a>, D: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a></span>","impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> for <a class=\"struct\" href=\"ndarray/struct.AxisIter.html\" title=\"struct ndarray::AxisIter\">AxisIter</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a>, D: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a></span>","impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> for <a class=\"struct\" href=\"ndarray/struct.AxisChunksIter.html\" title=\"struct ndarray::AxisChunksIter\">AxisChunksIter</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a>, D: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a></span>","impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> for <a class=\"struct\" href=\"ndarray/struct.IterMut.html\" title=\"struct ndarray::IterMut\">IterMut</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a>, D: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a></span>","impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> for <a class=\"struct\" href=\"ndarray/struct.IndexedIterMut.html\" title=\"struct ndarray::IndexedIterMut\">IndexedIterMut</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a>, D: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a></span>","impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> for <a class=\"struct\" href=\"ndarray/struct.InnerIterMut.html\" title=\"struct ndarray::InnerIterMut\">InnerIterMut</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a>, D: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a></span>","impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> for <a class=\"struct\" href=\"ndarray/struct.AxisIterMut.html\" title=\"struct ndarray::AxisIterMut\">AxisIterMut</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a>, D: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a></span>","impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a> for <a class=\"struct\" href=\"ndarray/struct.AxisChunksIterMut.html\" title=\"struct ndarray::AxisChunksIterMut\">AxisChunksIterMut</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a>, D: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a></span>",];

            if (window.register_implementors) {
                window.register_implementors(implementors);
            } else {
                window.pending_implementors = implementors;
            }
        
})()
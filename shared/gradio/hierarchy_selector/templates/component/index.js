const {
  SvelteComponent: Ll,
  append: T,
  attr: v,
  check_outros: Zt,
  create_component: Bl,
  destroy_block: Dl,
  destroy_component: Vl,
  detach: re,
  element: ue,
  empty: qt,
  ensure_array_like: je,
  flush: J,
  group_outros: jt,
  init: Cl,
  insert: ne,
  listen: Se,
  mount_component: El,
  noop: Nl,
  outro_and_destroy_block: Pl,
  run_all: xe,
  safe_not_equal: Tl,
  set_data: Ot,
  set_style: Qe,
  space: le,
  stop_propagation: bt,
  svg_element: ee,
  text: Gt,
  toggle_class: R,
  transition_in: Ee,
  transition_out: We,
  update_keyed_each: gt
} = window.__gradio__svelte__internal;
function pt(l, e, t) {
  const r = l.slice();
  r[18] = e[t];
  const n = (
    /*value*/
    r[4].includes(
      /*valueForItem*/
      r[10](
        /*item*/
        r[18]
      )
    )
  );
  return r[19] = n, r;
}
function vt(l, e, t) {
  const r = l.slice();
  r[22] = e[t];
  const n = Kt(
    /*folder*/
    r[22]
  );
  r[23] = n;
  const s = (
    /*valueForFolder*/
    r[11](
      /*folder*/
      r[22]
    )
  );
  r[24] = s;
  const o = (
    /*folderValue*/
    r[24] !== "" && /*value*/
    r[4].includes(
      /*folderValue*/
      r[24]
    )
  );
  return r[19] = o, r;
}
function Al(l) {
  let e, t, r, n, s, o, a, d, k;
  function w() {
    return (
      /*click_handler*/
      l[13](
        /*folder*/
        l[22]
      )
    );
  }
  return {
    c() {
      e = ue("button"), t = ee("svg"), r = ee("path"), v(r, "d", "M3.5 8.3l3 3 6-6.6"), v(t, "viewBox", "0 0 16 16"), v(t, "aria-hidden", "true"), v(e, "type", "button"), v(e, "class", "hierarchy-check"), v(e, "aria-pressed", n = /*selected*/
      l[19]), v(e, "aria-disabled", s = /*covered*/
      l[5] && !/*selected*/
      l[19]), v(e, "aria-label", o = `Select all of ${he(
        /*folder*/
        l[22]
      )}`), v(e, "title", a = /*covered*/
      l[5] && !/*selected*/
      l[19] ? "Included by a selected folder" : `Select all of ${he(
        /*folder*/
        l[22]
      )}`), R(
        e,
        "hierarchy-check-on",
        /*selected*/
        l[19]
      ), R(
        e,
        "hierarchy-check-covered",
        /*covered*/
        l[5] && !/*selected*/
        l[19]
      );
    },
    m(_, c) {
      ne(_, e, c), T(e, t), T(t, r), d || (k = [
        Se(e, "click", bt(w)),
        Se(e, "keydown", bt(Zl))
      ], d = !0);
    },
    p(_, c) {
      l = _, c & /*valueForFolder, folders, value*/
      2065 && n !== (n = /*selected*/
      l[19]) && v(e, "aria-pressed", n), c & /*covered, valueForFolder, folders, value*/
      2097 && s !== (s = /*covered*/
      l[5] && !/*selected*/
      l[19]) && v(e, "aria-disabled", s), c & /*folders*/
      1 && o !== (o = `Select all of ${he(
        /*folder*/
        l[22]
      )}`) && v(e, "aria-label", o), c & /*covered, valueForFolder, folders, value*/
      2097 && a !== (a = /*covered*/
      l[5] && !/*selected*/
      l[19] ? "Included by a selected folder" : `Select all of ${he(
        /*folder*/
        l[22]
      )}`) && v(e, "title", a), c & /*valueForFolder, folders, value*/
      2065 && R(
        e,
        "hierarchy-check-on",
        /*selected*/
        l[19]
      ), c & /*covered, valueForFolder, folders, value*/
      2097 && R(
        e,
        "hierarchy-check-covered",
        /*covered*/
        l[5] && !/*selected*/
        l[19]
      );
    },
    d(_) {
      _ && re(e), d = !1, xe(k);
    }
  };
}
function zl(l) {
  let e;
  return {
    c() {
      e = ue("span");
    },
    m(t, r) {
      ne(t, e, r);
    },
    p: Nl,
    d(t) {
      t && re(e);
    }
  };
}
function kt(l) {
  let e, t;
  return e = new Qt({
    props: {
      folders: (
        /*folder*/
        l[22].folders || []
      ),
      items: (
        /*folder*/
        l[22].items || []
      ),
      depth: (
        /*depth*/
        l[2] + 1
      ),
      expanded: (
        /*expanded*/
        l[3]
      ),
      value: (
        /*value*/
        l[4]
      ),
      covered: (
        /*covered*/
        l[5] || /*selected*/
        l[19]
      ),
      checkboxes: (
        /*checkboxes*/
        l[6]
      ),
      toggleItem: (
        /*toggleItem*/
        l[7]
      ),
      toggleFolder: (
        /*toggleFolder*/
        l[8]
      ),
      toggleFolderSelection: (
        /*toggleFolderSelection*/
        l[9]
      ),
      valueForItem: (
        /*valueForItem*/
        l[10]
      ),
      valueForFolder: (
        /*valueForFolder*/
        l[11]
      ),
      labelForItem: (
        /*labelForItem*/
        l[12]
      )
    }
  }), {
    c() {
      Bl(e.$$.fragment);
    },
    m(r, n) {
      El(e, r, n), t = !0;
    },
    p(r, n) {
      const s = {};
      n & /*folders*/
      1 && (s.folders = /*folder*/
      r[22].folders || []), n & /*folders*/
      1 && (s.items = /*folder*/
      r[22].items || []), n & /*depth*/
      4 && (s.depth = /*depth*/
      r[2] + 1), n & /*expanded*/
      8 && (s.expanded = /*expanded*/
      r[3]), n & /*value*/
      16 && (s.value = /*value*/
      r[4]), n & /*covered, valueForFolder, folders, value*/
      2097 && (s.covered = /*covered*/
      r[5] || /*selected*/
      r[19]), n & /*checkboxes*/
      64 && (s.checkboxes = /*checkboxes*/
      r[6]), n & /*toggleItem*/
      128 && (s.toggleItem = /*toggleItem*/
      r[7]), n & /*toggleFolder*/
      256 && (s.toggleFolder = /*toggleFolder*/
      r[8]), n & /*toggleFolderSelection*/
      512 && (s.toggleFolderSelection = /*toggleFolderSelection*/
      r[9]), n & /*valueForItem*/
      1024 && (s.valueForItem = /*valueForItem*/
      r[10]), n & /*valueForFolder*/
      2048 && (s.valueForFolder = /*valueForFolder*/
      r[11]), n & /*labelForItem*/
      4096 && (s.labelForItem = /*labelForItem*/
      r[12]), e.$set(s);
    },
    i(r) {
      t || (Ee(e.$$.fragment, r), t = !0);
    },
    o(r) {
      We(e.$$.fragment, r), t = !1;
    },
    d(r) {
      Vl(e, r);
    }
  };
}
function yt(l, e) {
  let t, r, n, s, o, a, d, k, w, _, c = he(
    /*folder*/
    e[22]
  ) + "", h, F, C, M = (
    /*expanded*/
    e[3].has(
      /*path*/
      e[23]
    )
  ), I, S, E, N;
  function m(y, L) {
    if (
      /*checkboxes*/
      y[6] && /*folderValue*/
      y[24] === ""
    ) return zl;
    if (
      /*checkboxes*/
      y[6]
    ) return Al;
  }
  let b = m(e), p = b && b(e);
  function P() {
    return (
      /*click_handler_1*/
      e[14](
        /*path*/
        e[23]
      )
    );
  }
  function z(...y) {
    return (
      /*keydown_handler_1*/
      e[15](
        /*path*/
        e[23],
        ...y
      )
    );
  }
  let f = M && kt(e);
  return {
    key: l,
    first: null,
    c() {
      t = ue("div"), r = ee("svg"), n = ee("path"), s = le(), p && p.c(), o = le(), a = ee("svg"), d = ee("path"), k = ee("path"), w = le(), _ = ue("span"), h = Gt(c), C = le(), f && f.c(), I = qt(), v(n, "d", "M6 4.5L10 8l-4 3.5"), v(r, "class", "hierarchy-twist svelte-t2420k"), v(r, "viewBox", "0 0 16 16"), v(r, "aria-hidden", "true"), R(
        r,
        "hierarchy-twist-open",
        /*expanded*/
        e[3].has(
          /*path*/
          e[23]
        )
      ), v(d, "d", "M2.75 6.25h5.4l1.55 1.7h7.55c.55 0 1 .45 1 1v6.3c0 .55-.45 1-1 1H2.75c-.55 0-1-.45-1-1v-8c0-.55.45-1 1-1Z"), v(k, "d", "M2.25 7.95V5.6c0-.55.45-1 1-1h4.5l1.35 1.65"), v(a, "class", "hierarchy-icon hierarchy-folder-icon svelte-t2420k"), v(a, "viewBox", "0 0 20 20"), v(a, "aria-hidden", "true"), v(_, "class", "hierarchy-name svelte-t2420k"), v(t, "class", "hierarchy-row hierarchy-folder svelte-t2420k"), v(t, "role", "button"), v(t, "tabindex", "0"), v(t, "title", F = he(
        /*folder*/
        e[22]
      )), R(
        t,
        "hierarchy-row-check",
        /*checkboxes*/
        e[6]
      ), R(
        t,
        "hierarchy-folder-selected",
        /*selected*/
        e[19]
      ), Qe(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), this.first = t;
    },
    m(y, L) {
      ne(y, t, L), T(t, r), T(r, n), T(t, s), p && p.m(t, null), T(t, o), T(t, a), T(a, d), T(a, k), T(t, w), T(t, _), T(_, h), ne(y, C, L), f && f.m(y, L), ne(y, I, L), S = !0, E || (N = [
        Se(t, "click", P),
        Se(t, "keydown", z)
      ], E = !0);
    },
    p(y, L) {
      e = y, (!S || L & /*expanded, folderPath, folders*/
      9) && R(
        r,
        "hierarchy-twist-open",
        /*expanded*/
        e[3].has(
          /*path*/
          e[23]
        )
      ), b === (b = m(e)) && p ? p.p(e, L) : (p && p.d(1), p = b && b(e), p && (p.c(), p.m(t, o))), (!S || L & /*folders*/
      1) && c !== (c = he(
        /*folder*/
        e[22]
      ) + "") && Ot(h, c), (!S || L & /*folders*/
      1 && F !== (F = he(
        /*folder*/
        e[22]
      ))) && v(t, "title", F), (!S || L & /*checkboxes*/
      64) && R(
        t,
        "hierarchy-row-check",
        /*checkboxes*/
        e[6]
      ), (!S || L & /*valueForFolder, folders, value*/
      2065) && R(
        t,
        "hierarchy-folder-selected",
        /*selected*/
        e[19]
      ), L & /*depth*/
      4 && Qe(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), L & /*expanded, folders*/
      9 && (M = /*expanded*/
      e[3].has(
        /*path*/
        e[23]
      )), M ? f ? (f.p(e, L), L & /*expanded, folders*/
      9 && Ee(f, 1)) : (f = kt(e), f.c(), Ee(f, 1), f.m(I.parentNode, I)) : f && (jt(), We(f, 1, 1, () => {
        f = null;
      }), Zt());
    },
    i(y) {
      S || (Ee(f), S = !0);
    },
    o(y) {
      We(f), S = !1;
    },
    d(y) {
      y && (re(t), re(C), re(I)), p && p.d(), f && f.d(y), E = !1, xe(N);
    }
  };
}
function wt(l) {
  let e;
  return {
    c() {
      e = ue("span"), e.innerHTML = '<svg viewBox="0 0 16 16"><path d="M3.5 8.3l3 3 6-6.6"></path></svg>', v(e, "class", "hierarchy-check"), v(e, "aria-hidden", "true"), R(
        e,
        "hierarchy-check-on",
        /*selected*/
        l[19]
      ), R(
        e,
        "hierarchy-check-covered",
        /*covered*/
        l[5] && !/*selected*/
        l[19]
      );
    },
    m(t, r) {
      ne(t, e, r);
    },
    p(t, r) {
      r & /*value, valueForItem, items*/
      1042 && R(
        e,
        "hierarchy-check-on",
        /*selected*/
        t[19]
      ), r & /*covered, value, valueForItem, items*/
      1074 && R(
        e,
        "hierarchy-check-covered",
        /*covered*/
        t[5] && !/*selected*/
        t[19]
      );
    },
    d(t) {
      t && re(e);
    }
  };
}
function Ft(l, e) {
  let t, r, n, s, o, a, d, k, w, _ = (
    /*labelForItem*/
    e[12](
      /*item*/
      e[18]
    ) + ""
  ), c, h, F, C, M, I, S = (
    /*checkboxes*/
    e[6] && wt(e)
  );
  function E() {
    return (
      /*click_handler_2*/
      e[16](
        /*item*/
        e[18]
      )
    );
  }
  function N(...m) {
    return (
      /*keydown_handler_2*/
      e[17](
        /*item*/
        e[18],
        ...m
      )
    );
  }
  return {
    key: l,
    first: null,
    c() {
      t = ue("div"), r = ue("span"), n = le(), S && S.c(), s = le(), o = ee("svg"), a = ee("path"), d = ee("path"), k = le(), w = ue("span"), c = Gt(_), h = le(), v(r, "class", "hierarchy-twist-spacer svelte-t2420k"), v(a, "d", "M5.25 2.75h6.05L15.75 7.2v10.05H5.25V2.75Z"), v(d, "d", "M11.25 2.95V7.3h4.3"), v(o, "class", "hierarchy-icon hierarchy-item-icon svelte-t2420k"), v(o, "viewBox", "0 0 20 20"), v(o, "aria-hidden", "true"), v(w, "class", "hierarchy-name svelte-t2420k"), v(t, "class", "hierarchy-row hierarchy-item svelte-t2420k"), v(t, "title", F = /*covered*/
      e[5] && !/*selected*/
      e[19] ? "Included by a selected folder" : (
        /*labelForItem*/
        e[12](
          /*item*/
          e[18]
        )
      )), v(t, "role", "button"), v(t, "tabindex", "0"), v(t, "aria-pressed", C = /*selected*/
      e[19]), R(
        t,
        "hierarchy-row-check",
        /*checkboxes*/
        e[6]
      ), R(
        t,
        "hierarchy-item-selected",
        /*selected*/
        e[19]
      ), R(
        t,
        "hierarchy-item-covered",
        /*covered*/
        e[5] && !/*selected*/
        e[19]
      ), Qe(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`), this.first = t;
    },
    m(m, b) {
      ne(m, t, b), T(t, r), T(t, n), S && S.m(t, null), T(t, s), T(t, o), T(o, a), T(o, d), T(t, k), T(t, w), T(w, c), T(t, h), M || (I = [
        Se(t, "click", E),
        Se(t, "keydown", N)
      ], M = !0);
    },
    p(m, b) {
      e = m, /*checkboxes*/
      e[6] ? S ? S.p(e, b) : (S = wt(e), S.c(), S.m(t, s)) : S && (S.d(1), S = null), b & /*labelForItem, items*/
      4098 && _ !== (_ = /*labelForItem*/
      e[12](
        /*item*/
        e[18]
      ) + "") && Ot(c, _), b & /*covered, value, valueForItem, items, labelForItem*/
      5170 && F !== (F = /*covered*/
      e[5] && !/*selected*/
      e[19] ? "Included by a selected folder" : (
        /*labelForItem*/
        e[12](
          /*item*/
          e[18]
        )
      )) && v(t, "title", F), b & /*value, valueForItem, items*/
      1042 && C !== (C = /*selected*/
      e[19]) && v(t, "aria-pressed", C), b & /*checkboxes*/
      64 && R(
        t,
        "hierarchy-row-check",
        /*checkboxes*/
        e[6]
      ), b & /*value, valueForItem, items*/
      1042 && R(
        t,
        "hierarchy-item-selected",
        /*selected*/
        e[19]
      ), b & /*covered, value, valueForItem, items*/
      1074 && R(
        t,
        "hierarchy-item-covered",
        /*covered*/
        e[5] && !/*selected*/
        e[19]
      ), b & /*depth*/
      4 && Qe(t, "padding-left", `${/*depth*/
      e[2] * 18 + 6}px`);
    },
    d(m) {
      m && re(t), S && S.d(), M = !1, xe(I);
    }
  };
}
function Rl(l) {
  let e = [], t = /* @__PURE__ */ new Map(), r, n = [], s = /* @__PURE__ */ new Map(), o, a, d = je(
    /*folders*/
    l[0]
  );
  const k = (c) => Kt(
    /*folder*/
    c[22]
  );
  for (let c = 0; c < d.length; c += 1) {
    let h = vt(l, d, c), F = k(h);
    t.set(F, e[c] = yt(F, h));
  }
  let w = je(
    /*items*/
    l[1]
  );
  const _ = (c) => (
    /*valueForItem*/
    c[10](
      /*item*/
      c[18]
    )
  );
  for (let c = 0; c < w.length; c += 1) {
    let h = pt(l, w, c), F = _(h);
    s.set(F, n[c] = Ft(F, h));
  }
  return {
    c() {
      for (let c = 0; c < e.length; c += 1)
        e[c].c();
      r = le();
      for (let c = 0; c < n.length; c += 1)
        n[c].c();
      o = qt();
    },
    m(c, h) {
      for (let F = 0; F < e.length; F += 1)
        e[F] && e[F].m(c, h);
      ne(c, r, h);
      for (let F = 0; F < n.length; F += 1)
        n[F] && n[F].m(c, h);
      ne(c, o, h), a = !0;
    },
    p(c, [h]) {
      h & /*folders, depth, expanded, value, covered, valueForFolder, checkboxes, toggleItem, toggleFolder, toggleFolderSelection, valueForItem, labelForItem, folderPath, folderLabel*/
      8189 && (d = je(
        /*folders*/
        c[0]
      ), jt(), e = gt(e, h, k, 1, c, d, t, r.parentNode, Pl, yt, r, vt), Zt()), h & /*covered, value, valueForItem, items, labelForItem, checkboxes, depth, toggleItem*/
      5366 && (w = je(
        /*items*/
        c[1]
      ), n = gt(n, h, _, 1, c, w, s, o.parentNode, Dl, Ft, o, pt));
    },
    i(c) {
      if (!a) {
        for (let h = 0; h < d.length; h += 1)
          Ee(e[h]);
        a = !0;
      }
    },
    o(c) {
      for (let h = 0; h < e.length; h += 1)
        We(e[h]);
      a = !1;
    },
    d(c) {
      c && (re(r), re(o));
      for (let h = 0; h < e.length; h += 1)
        e[h].d(c);
      for (let h = 0; h < n.length; h += 1)
        n[h].d(c);
    }
  };
}
function he(l) {
  return String(l.name || l.path || "");
}
function Kt(l) {
  return String(l.path || l.name || "");
}
const Zl = () => {
};
function ql(l, e, t) {
  let { folders: r = [] } = e, { items: n = [] } = e, { depth: s = 0 } = e, { expanded: o } = e, { value: a } = e, { covered: d = !1 } = e, { checkboxes: k = !1 } = e, { toggleItem: w } = e, { toggleFolder: _ } = e, { toggleFolderSelection: c } = e, { valueForItem: h } = e, { valueForFolder: F } = e, { labelForItem: C } = e;
  const M = (m) => c(m, d), I = (m) => _(m), S = (m, b) => {
    (b.key === "Enter" || b.key === " ") && (b.preventDefault(), _(m));
  }, E = (m) => w(m, d), N = (m, b) => {
    (b.key === "Enter" || b.key === " ") && (b.preventDefault(), w(m, d));
  };
  return l.$$set = (m) => {
    "folders" in m && t(0, r = m.folders), "items" in m && t(1, n = m.items), "depth" in m && t(2, s = m.depth), "expanded" in m && t(3, o = m.expanded), "value" in m && t(4, a = m.value), "covered" in m && t(5, d = m.covered), "checkboxes" in m && t(6, k = m.checkboxes), "toggleItem" in m && t(7, w = m.toggleItem), "toggleFolder" in m && t(8, _ = m.toggleFolder), "toggleFolderSelection" in m && t(9, c = m.toggleFolderSelection), "valueForItem" in m && t(10, h = m.valueForItem), "valueForFolder" in m && t(11, F = m.valueForFolder), "labelForItem" in m && t(12, C = m.labelForItem);
  }, [
    r,
    n,
    s,
    o,
    a,
    d,
    k,
    w,
    _,
    c,
    h,
    F,
    C,
    M,
    I,
    S,
    E,
    N
  ];
}
class Qt extends Ll {
  constructor(e) {
    super(), Cl(this, e, ql, Rl, Tl, {
      folders: 0,
      items: 1,
      depth: 2,
      expanded: 3,
      value: 4,
      covered: 5,
      checkboxes: 6,
      toggleItem: 7,
      toggleFolder: 8,
      toggleFolderSelection: 9,
      valueForItem: 10,
      valueForFolder: 11,
      labelForItem: 12
    });
  }
  get folders() {
    return this.$$.ctx[0];
  }
  set folders(e) {
    this.$$set({ folders: e }), J();
  }
  get items() {
    return this.$$.ctx[1];
  }
  set items(e) {
    this.$$set({ items: e }), J();
  }
  get depth() {
    return this.$$.ctx[2];
  }
  set depth(e) {
    this.$$set({ depth: e }), J();
  }
  get expanded() {
    return this.$$.ctx[3];
  }
  set expanded(e) {
    this.$$set({ expanded: e }), J();
  }
  get value() {
    return this.$$.ctx[4];
  }
  set value(e) {
    this.$$set({ value: e }), J();
  }
  get covered() {
    return this.$$.ctx[5];
  }
  set covered(e) {
    this.$$set({ covered: e }), J();
  }
  get checkboxes() {
    return this.$$.ctx[6];
  }
  set checkboxes(e) {
    this.$$set({ checkboxes: e }), J();
  }
  get toggleItem() {
    return this.$$.ctx[7];
  }
  set toggleItem(e) {
    this.$$set({ toggleItem: e }), J();
  }
  get toggleFolder() {
    return this.$$.ctx[8];
  }
  set toggleFolder(e) {
    this.$$set({ toggleFolder: e }), J();
  }
  get toggleFolderSelection() {
    return this.$$.ctx[9];
  }
  set toggleFolderSelection(e) {
    this.$$set({ toggleFolderSelection: e }), J();
  }
  get valueForItem() {
    return this.$$.ctx[10];
  }
  set valueForItem(e) {
    this.$$set({ valueForItem: e }), J();
  }
  get valueForFolder() {
    return this.$$.ctx[11];
  }
  set valueForFolder(e) {
    this.$$set({ valueForFolder: e }), J();
  }
  get labelForItem() {
    return this.$$.ctx[12];
  }
  set labelForItem(e) {
    this.$$set({ labelForItem: e }), J();
  }
}
const {
  SvelteComponent: jl,
  action_destroyer: Ol,
  append: B,
  attr: u,
  binding_callbacks: De,
  check_outros: et,
  create_component: Gl,
  destroy_block: Kl,
  destroy_component: Ql,
  destroy_each: Wl,
  detach: Z,
  element: A,
  empty: tt,
  ensure_array_like: Je,
  flush: j,
  group_outros: lt,
  init: Jl,
  insert: q,
  listen: W,
  mount_component: Ul,
  noop: $e,
  null_to_empty: It,
  run_all: rt,
  safe_not_equal: Xl,
  set_data: ve,
  set_input_value: St,
  space: U,
  stop_propagation: nt,
  svg_element: ge,
  text: ie,
  toggle_class: K,
  transition_in: se,
  transition_out: pe,
  update_keyed_each: Yl
} = window.__gradio__svelte__internal, { onDestroy: $l, onMount: xl, tick: Oe } = window.__gradio__svelte__internal;
function Mt(l, e, t) {
  const r = l.slice();
  r[92] = e[t];
  const n = (
    /*selectedValue*/
    r[14].includes(
      /*item*/
      r[92].search_folder ? be(
        /*item*/
        r[92]
      ) : Ie(
        /*item*/
        r[92]
      )
    )
  );
  r[93] = n;
  const s = Xt(
    /*item*/
    r[92],
    /*selectedValue*/
    r[14]
  );
  return r[94] = s, r;
}
function Ht(l, e, t) {
  const r = l.slice();
  return r[93] = e[t], r[98] = t, r;
}
function Lt(l) {
  let e, t, r, n, s, o, a, d, k, w, _, c, h, F, C, M, I, S, E, N = (
    /*show_label*/
    l[6] && /*label*/
    l[4] && Bt(l)
  ), m = Je(
    /*selectedValue*/
    l[14]
  ), b = [];
  for (let f = 0; f < m.length; f += 1)
    b[f] = Vt(Ht(l, m, f));
  let p = (
    /*selectedValue*/
    l[14].length > 0 && Ct(l)
  ), P = (
    /*open*/
    l[8] && Et(l)
  ), z = (
    /*info*/
    l[5] && At(l)
  );
  return {
    c() {
      e = A("div"), t = A("div"), N && N.c(), r = U(), n = A("div"), s = A("div");
      for (let f = 0; f < b.length; f += 1)
        b[f].c();
      o = U(), a = A("input"), _ = U(), p && p.c(), F = U(), P && P.c(), C = U(), z && z.c(), u(a, "class", "hierarchy-selector-search-input svelte-8x1vbd"), u(a, "type", "text"), u(a, "autocomplete", "off"), u(a, "spellcheck", "false"), a.disabled = d = !/*interactive*/
      l[7], u(a, "tabindex", k = /*interactive*/
      l[7] ? 0 : -1), u(a, "placeholder", w = /*show_placeholder*/
      l[3] && /*selectedValue*/
      l[14].length === 0 ? (
        /*label*/
        l[4]
      ) : ""), u(
        a,
        "aria-label",
        /*label*/
        l[4]
      ), u(s, "class", "hierarchy-selector-chips svelte-8x1vbd"), u(n, "class", "hierarchy-selector-input svelte-8x1vbd"), u(n, "role", "combobox"), u(n, "tabindex", c = /*interactive*/
      l[7] ? 0 : -1), u(n, "aria-haspopup", "tree"), u(n, "aria-expanded", h = /*open*/
      l[8] ? "true" : "false"), u(
        n,
        "aria-controls",
        /*panelId*/
        l[27]
      ), K(n, "hierarchy-selector-disabled", !/*interactive*/
      l[7]), u(t, "class", "hierarchy-selector-field svelte-8x1vbd"), u(
        e,
        "id",
        /*elem_id*/
        l[0]
      ), u(e, "class", M = It(
        /*classes*/
        l[26]
      ) + " svelte-8x1vbd"), u(
        e,
        "style",
        /*style*/
        l[25]
      );
    },
    m(f, y) {
      q(f, e, y), B(e, t), N && N.m(t, null), B(t, r), B(t, n), B(n, s);
      for (let L = 0; L < b.length; L += 1)
        b[L] && b[L].m(s, null);
      B(s, o), B(s, a), l[68](a), St(
        a,
        /*searchQuery*/
        l[10]
      ), B(n, _), p && p.m(n, null), l[70](n), B(t, F), P && P.m(t, null), B(e, C), z && z.m(e, null), l[75](e), I = !0, S || (E = [
        W(
          a,
          "input",
          /*input_input_handler*/
          l[69]
        ),
        W(
          a,
          "focus",
          /*onSearchFocus*/
          l[37]
        ),
        W(
          a,
          "input",
          /*onSearchInput*/
          l[38]
        ),
        W(a, "keydown", nt(
          /*onInputKeydown*/
          l[43]
        )),
        W(
          n,
          "mousedown",
          /*onInputPointerDown*/
          l[39]
        ),
        W(
          n,
          "keydown",
          /*onInputKeydown*/
          l[43]
        )
      ], S = !0);
    },
    p(f, y) {
      if (/*show_label*/
      f[6] && /*label*/
      f[4] ? N ? N.p(f, y) : (N = Bt(f), N.c(), N.m(t, r)) : N && (N.d(1), N = null), y[0] & /*interactive, draggedIndex, dragOverIndex, displayValue, selectedValue, folderValues*/
      1086343296 | y[1] & /*onDragStart, onDragEnd, onDrop, removeValue*/
      3592) {
        m = Je(
          /*selectedValue*/
          f[14]
        );
        let L;
        for (L = 0; L < m.length; L += 1) {
          const Me = Ht(f, m, L);
          b[L] ? b[L].p(Me, y) : (b[L] = Vt(Me), b[L].c(), b[L].m(s, o));
        }
        for (; L < b.length; L += 1)
          b[L].d(1);
        b.length = m.length;
      }
      (!I || y[0] & /*interactive*/
      128 && d !== (d = !/*interactive*/
      f[7])) && (a.disabled = d), (!I || y[0] & /*interactive*/
      128 && k !== (k = /*interactive*/
      f[7] ? 0 : -1)) && u(a, "tabindex", k), (!I || y[0] & /*show_placeholder, selectedValue, label*/
      16408 && w !== (w = /*show_placeholder*/
      f[3] && /*selectedValue*/
      f[14].length === 0 ? (
        /*label*/
        f[4]
      ) : "")) && u(a, "placeholder", w), (!I || y[0] & /*label*/
      16) && u(
        a,
        "aria-label",
        /*label*/
        f[4]
      ), y[0] & /*searchQuery*/
      1024 && a.value !== /*searchQuery*/
      f[10] && St(
        a,
        /*searchQuery*/
        f[10]
      ), /*selectedValue*/
      f[14].length > 0 ? p ? p.p(f, y) : (p = Ct(f), p.c(), p.m(n, null)) : p && (p.d(1), p = null), (!I || y[0] & /*interactive*/
      128 && c !== (c = /*interactive*/
      f[7] ? 0 : -1)) && u(n, "tabindex", c), (!I || y[0] & /*open*/
      256 && h !== (h = /*open*/
      f[8] ? "true" : "false")) && u(n, "aria-expanded", h), (!I || y[0] & /*panelId*/
      134217728) && u(
        n,
        "aria-controls",
        /*panelId*/
        f[27]
      ), (!I || y[0] & /*interactive*/
      128) && K(n, "hierarchy-selector-disabled", !/*interactive*/
      f[7]), /*open*/
      f[8] ? P ? (P.p(f, y), y[0] & /*open*/
      256 && se(P, 1)) : (P = Et(f), P.c(), se(P, 1), P.m(t, null)) : P && (lt(), pe(P, 1, 1, () => {
        P = null;
      }), et()), /*info*/
      f[5] ? z ? z.p(f, y) : (z = At(f), z.c(), z.m(e, null)) : z && (z.d(1), z = null), (!I || y[0] & /*elem_id*/
      1) && u(
        e,
        "id",
        /*elem_id*/
        f[0]
      ), (!I || y[0] & /*classes*/
      67108864 && M !== (M = It(
        /*classes*/
        f[26]
      ) + " svelte-8x1vbd")) && u(e, "class", M), (!I || y[0] & /*style*/
      33554432) && u(
        e,
        "style",
        /*style*/
        f[25]
      );
    },
    i(f) {
      I || (se(P), I = !0);
    },
    o(f) {
      pe(P), I = !1;
    },
    d(f) {
      f && Z(e), N && N.d(), Wl(b, f), l[68](null), p && p.d(), l[70](null), P && P.d(), z && z.d(), l[75](null), S = !1, rt(E);
    }
  };
}
function Bt(l) {
  let e, t;
  return {
    c() {
      e = A("span"), t = ie(
        /*label*/
        l[4]
      ), u(e, "class", "hierarchy-selector-label svelte-8x1vbd");
    },
    m(r, n) {
      q(r, e, n), B(e, t);
    },
    p(r, n) {
      n[0] & /*label*/
      16 && ve(
        t,
        /*label*/
        r[4]
      );
    },
    d(r) {
      r && Z(e);
    }
  };
}
function Dt(l) {
  let e, t;
  return {
    c() {
      e = ge("svg"), t = ge("path"), u(t, "d", "M2.75 6.25h5.4l1.55 1.7h7.55c.55 0 1 .45 1 1v6.3c0 .55-.45 1-1 1H2.75c-.55 0-1-.45-1-1v-8c0-.55.45-1 1-1Z"), u(e, "class", "hierarchy-selector-chip-folder svelte-8x1vbd"), u(e, "viewBox", "0 0 20 20"), u(e, "aria-hidden", "true");
    },
    m(r, n) {
      q(r, e, n), B(e, t);
    },
    d(r) {
      r && Z(e);
    }
  };
}
function Vt(l) {
  let e, t = (
    /*folderValues*/
    l[11].has(
      /*selected*/
      l[93]
    )
  ), r, n, s = (
    /*displayValue*/
    l[30](
      /*selected*/
      l[93]
    ) + ""
  ), o, a, d, k, w, _ = t && Dt();
  function c() {
    return (
      /*click_handler*/
      l[63](
        /*index*/
        l[98]
      )
    );
  }
  function h(...M) {
    return (
      /*dragstart_handler*/
      l[64](
        /*index*/
        l[98],
        ...M
      )
    );
  }
  function F(...M) {
    return (
      /*dragover_handler*/
      l[65](
        /*index*/
        l[98],
        ...M
      )
    );
  }
  function C(...M) {
    return (
      /*drop_handler*/
      l[67](
        /*index*/
        l[98],
        ...M
      )
    );
  }
  return {
    c() {
      e = A("span"), _ && _.c(), r = U(), n = A("span"), o = ie(s), a = U(), d = A("button"), d.textContent = "x", u(n, "class", "hierarchy-selector-chip-text svelte-8x1vbd"), u(d, "type", "button"), u(d, "class", "hierarchy-selector-remove svelte-8x1vbd"), u(d, "aria-label", "Remove"), u(e, "class", "hierarchy-selector-chip svelte-8x1vbd"), u(e, "role", "listitem"), u(
        e,
        "draggable",
        /*interactive*/
        l[7]
      ), K(
        e,
        "hierarchy-selector-chip-dragging",
        /*draggedIndex*/
        l[22] === /*index*/
        l[98]
      ), K(
        e,
        "hierarchy-selector-chip-over",
        /*dragOverIndex*/
        l[23] === /*index*/
        l[98]
      );
    },
    m(M, I) {
      q(M, e, I), _ && _.m(e, null), B(e, r), B(e, n), B(n, o), B(e, a), B(e, d), k || (w = [
        W(d, "click", nt(c)),
        W(e, "dragstart", h),
        W(
          e,
          "dragend",
          /*onDragEnd*/
          l[41]
        ),
        W(e, "dragover", F),
        W(
          e,
          "dragleave",
          /*dragleave_handler*/
          l[66]
        ),
        W(e, "drop", C)
      ], k = !0);
    },
    p(M, I) {
      l = M, I[0] & /*folderValues, selectedValue*/
      18432 && (t = /*folderValues*/
      l[11].has(
        /*selected*/
        l[93]
      )), t ? _ || (_ = Dt(), _.c(), _.m(e, r)) : _ && (_.d(1), _ = null), I[0] & /*selectedValue*/
      16384 && s !== (s = /*displayValue*/
      l[30](
        /*selected*/
        l[93]
      ) + "") && ve(o, s), I[0] & /*interactive*/
      128 && u(
        e,
        "draggable",
        /*interactive*/
        l[7]
      ), I[0] & /*draggedIndex*/
      4194304 && K(
        e,
        "hierarchy-selector-chip-dragging",
        /*draggedIndex*/
        l[22] === /*index*/
        l[98]
      ), I[0] & /*dragOverIndex*/
      8388608 && K(
        e,
        "hierarchy-selector-chip-over",
        /*dragOverIndex*/
        l[23] === /*index*/
        l[98]
      );
    },
    d(M) {
      M && Z(e), _ && _.d(), k = !1, rt(w);
    }
  };
}
function Ct(l) {
  let e, t, r;
  return {
    c() {
      e = A("button"), e.textContent = "x", u(e, "type", "button"), u(e, "class", "hierarchy-selector-clear svelte-8x1vbd"), u(e, "aria-label", "Clear selection");
    },
    m(n, s) {
      q(n, e, s), t || (r = W(e, "click", nt(
        /*clearValues*/
        l[35]
      )), t = !0);
    },
    p: $e,
    d(n) {
      n && Z(e), t = !1, r();
    }
  };
}
function Et(l) {
  let e, t, r, n, s, o, a;
  const d = [tr, er], k = [];
  function w(_, c) {
    return (
      /*searchMode*/
      _[15] ? 0 : 1
    );
  }
  return r = w(l), n = k[r] = d[r](l), {
    c() {
      e = A("div"), t = A("div"), n.c(), u(t, "class", "hierarchy-selector-panel-content svelte-8x1vbd"), u(
        e,
        "id",
        /*panelId*/
        l[27]
      ), u(e, "class", "hierarchy-selector-panel svelte-8x1vbd"), u(
        e,
        "style",
        /*panelStyle*/
        l[24]
      );
    },
    m(_, c) {
      q(_, e, c), B(e, t), k[r].m(t, null), l[73](t), l[74](e), s = !0, o || (a = Ol(_r.call(null, e)), o = !0);
    },
    p(_, c) {
      let h = r;
      r = w(_), r === h ? k[r].p(_, c) : (lt(), pe(k[h], 1, 1, () => {
        k[h] = null;
      }), et(), n = k[r], n ? n.p(_, c) : (n = k[r] = d[r](_), n.c()), se(n, 1), n.m(t, null)), (!s || c[0] & /*panelId*/
      134217728) && u(
        e,
        "id",
        /*panelId*/
        _[27]
      ), (!s || c[0] & /*panelStyle*/
      16777216) && u(
        e,
        "style",
        /*panelStyle*/
        _[24]
      );
    },
    i(_) {
      s || (se(n), s = !0);
    },
    o(_) {
      pe(n), s = !1;
    },
    d(_) {
      _ && Z(e), k[r].d(), l[73](null), l[74](null), o = !1, a();
    }
  };
}
function er(l) {
  let e, t;
  return e = new Qt({
    props: {
      folders: (
        /*normalizedHierarchy*/
        l[13].folders || []
      ),
      items: (
        /*normalizedHierarchy*/
        l[13].items || []
      ),
      depth: 0,
      expanded: (
        /*expanded*/
        l[9]
      ),
      value: (
        /*selectedValue*/
        l[14]
      ),
      checkboxes: (
        /*checkboxes*/
        l[28]
      ),
      toggleItem: (
        /*toggleItem*/
        l[31]
      ),
      toggleFolder: (
        /*toggleFolder*/
        l[36]
      ),
      toggleFolderSelection: (
        /*toggleFolderSelection*/
        l[32]
      ),
      valueForItem: Ie,
      valueForFolder: be,
      labelForItem: oe
    }
  }), {
    c() {
      Gl(e.$$.fragment);
    },
    m(r, n) {
      Ul(e, r, n), t = !0;
    },
    p(r, n) {
      const s = {};
      n[0] & /*normalizedHierarchy*/
      8192 && (s.folders = /*normalizedHierarchy*/
      r[13].folders || []), n[0] & /*normalizedHierarchy*/
      8192 && (s.items = /*normalizedHierarchy*/
      r[13].items || []), n[0] & /*expanded*/
      512 && (s.expanded = /*expanded*/
      r[9]), n[0] & /*selectedValue*/
      16384 && (s.value = /*selectedValue*/
      r[14]), n[0] & /*checkboxes*/
      268435456 && (s.checkboxes = /*checkboxes*/
      r[28]), e.$set(s);
    },
    i(r) {
      t || (se(e.$$.fragment, r), t = !0);
    },
    o(r) {
      pe(e.$$.fragment, r), t = !1;
    },
    d(r) {
      Ql(e, r);
    }
  };
}
function tr(l) {
  let e;
  function t(s, o) {
    return (
      /*searchResults*/
      s[12].length ? rr : lr
    );
  }
  let r = t(l), n = r(l);
  return {
    c() {
      n.c(), e = tt();
    },
    m(s, o) {
      n.m(s, o), q(s, e, o);
    },
    p(s, o) {
      r === (r = t(s)) && n ? n.p(s, o) : (n.d(1), n = r(s), n && (n.c(), n.m(e.parentNode, e)));
    },
    i: $e,
    o: $e,
    d(s) {
      s && Z(e), n.d(s);
    }
  };
}
function lr(l) {
  let e, t;
  return {
    c() {
      e = A("div"), t = ie(
        /*search_empty_label*/
        l[2]
      ), u(e, "class", "hierarchy-search-empty svelte-8x1vbd");
    },
    m(r, n) {
      q(r, e, n), B(e, t);
    },
    p(r, n) {
      n[0] & /*search_empty_label*/
      4 && ve(
        t,
        /*search_empty_label*/
        r[2]
      );
    },
    d(r) {
      r && Z(e);
    }
  };
}
function rr(l) {
  let e = [], t = /* @__PURE__ */ new Map(), r, n = Je(
    /*searchResults*/
    l[12]
  );
  const s = (o) => (
    /*item*/
    o[92].search_key
  );
  for (let o = 0; o < n.length; o += 1) {
    let a = Mt(l, n, o), d = s(a);
    t.set(d, e[o] = Tt(d, a));
  }
  return {
    c() {
      for (let o = 0; o < e.length; o += 1)
        e[o].c();
      r = tt();
    },
    m(o, a) {
      for (let d = 0; d < e.length; d += 1)
        e[d] && e[d].m(o, a);
      q(o, r, a);
    },
    p(o, a) {
      a[0] & /*searchResults, selectedValue, selectedLabelForItem, checkboxes, breadcrumbMode*/
      805392384 | a[1] & /*toggleEntry*/
      4 && (n = Je(
        /*searchResults*/
        o[12]
      ), e = Yl(e, a, s, 1, o, n, t, r.parentNode, Kl, Tt, r, Mt));
    },
    d(o) {
      o && Z(r);
      for (let a = 0; a < e.length; a += 1)
        e[a].d(o);
    }
  };
}
function Nt(l) {
  let e;
  return {
    c() {
      e = A("span"), e.innerHTML = '<svg viewBox="0 0 16 16"><path d="M3.5 8.3l3 3 6-6.6"></path></svg>', u(e, "class", "hierarchy-check"), u(e, "aria-hidden", "true"), K(
        e,
        "hierarchy-check-on",
        /*selected*/
        l[93]
      ), K(
        e,
        "hierarchy-check-covered",
        /*covered*/
        l[94] && !/*selected*/
        l[93]
      );
    },
    m(t, r) {
      q(t, e, r);
    },
    p(t, r) {
      r[0] & /*selectedValue, searchResults*/
      20480 && K(
        e,
        "hierarchy-check-on",
        /*selected*/
        t[93]
      ), r[0] & /*searchResults, selectedValue*/
      20480 && K(
        e,
        "hierarchy-check-covered",
        /*covered*/
        t[94] && !/*selected*/
        t[93]
      );
    },
    d(t) {
      t && Z(e);
    }
  };
}
function nr(l) {
  let e, t;
  return {
    c() {
      e = ge("path"), t = ge("path"), u(e, "d", "M5.25 2.75h6.05L15.75 7.2v10.05H5.25V2.75Z"), u(t, "d", "M11.25 2.95V7.3h4.3");
    },
    m(r, n) {
      q(r, e, n), q(r, t, n);
    },
    d(r) {
      r && (Z(e), Z(t));
    }
  };
}
function ir(l) {
  let e, t;
  return {
    c() {
      e = ge("path"), t = ge("path"), u(e, "d", "M2.75 6.25h5.4l1.55 1.7h7.55c.55 0 1 .45 1 1v6.3c0 .55-.45 1-1 1H2.75c-.55 0-1-.45-1-1v-8c0-.55.45-1 1-1Z"), u(t, "d", "M2.25 7.95V5.6c0-.55.45-1 1-1h4.5l1.35 1.65");
    },
    m(r, n) {
      q(r, e, n), q(r, t, n);
    },
    d(r) {
      r && (Z(e), Z(t));
    }
  };
}
function sr(l) {
  let e, t, r = oe(
    /*item*/
    l[92]
  ) + "", n, s, o = (
    /*item*/
    l[92].search_path && Pt(l)
  );
  return {
    c() {
      e = A("span"), t = A("span"), n = ie(r), s = U(), o && o.c(), u(t, "class", "hierarchy-search-name svelte-8x1vbd"), u(e, "class", "hierarchy-search-label svelte-8x1vbd");
    },
    m(a, d) {
      q(a, e, d), B(e, t), B(t, n), B(e, s), o && o.m(e, null);
    },
    p(a, d) {
      d[0] & /*searchResults*/
      4096 && r !== (r = oe(
        /*item*/
        a[92]
      ) + "") && ve(n, r), /*item*/
      a[92].search_path ? o ? o.p(a, d) : (o = Pt(a), o.c(), o.m(e, null)) : o && (o.d(1), o = null);
    },
    d(a) {
      a && Z(e), o && o.d();
    }
  };
}
function or(l) {
  let e, t = (
    /*item*/
    l[92].search_display + ""
  ), r;
  return {
    c() {
      e = A("span"), r = ie(t), u(e, "class", "hierarchy-search-label hierarchy-search-name svelte-8x1vbd");
    },
    m(n, s) {
      q(n, e, s), B(e, r);
    },
    p(n, s) {
      s[0] & /*searchResults*/
      4096 && t !== (t = /*item*/
      n[92].search_display + "") && ve(r, t);
    },
    d(n) {
      n && Z(e);
    }
  };
}
function Pt(l) {
  let e, t, r = (
    /*item*/
    l[92].search_path + ""
  ), n, s;
  return {
    c() {
      e = A("span"), t = ie("["), n = ie(r), s = ie("]"), u(e, "class", "hierarchy-search-path svelte-8x1vbd");
    },
    m(o, a) {
      q(o, e, a), B(e, t), B(e, n), B(e, s);
    },
    p(o, a) {
      a[0] & /*searchResults*/
      4096 && r !== (r = /*item*/
      o[92].search_path + "") && ve(n, r);
    },
    d(o) {
      o && Z(e);
    }
  };
}
function Tt(l, e) {
  let t, r, n, s, o, a, d, k, w, _, c, h = (
    /*checkboxes*/
    e[28] && Nt(e)
  );
  function F(b, p) {
    return (
      /*item*/
      b[92].search_folder ? ir : nr
    );
  }
  let C = F(e), M = C(e);
  function I(b, p) {
    return (
      /*breadcrumbMode*/
      b[16] ? or : sr
    );
  }
  let S = I(e), E = S(e);
  function N() {
    return (
      /*click_handler_1*/
      e[71](
        /*item*/
        e[92],
        /*covered*/
        e[94]
      )
    );
  }
  function m(...b) {
    return (
      /*keydown_handler*/
      e[72](
        /*item*/
        e[92],
        /*covered*/
        e[94],
        ...b
      )
    );
  }
  return {
    key: l,
    first: null,
    c() {
      t = A("div"), r = A("span"), n = U(), h && h.c(), s = U(), o = ge("svg"), M.c(), a = U(), E.c(), d = U(), u(r, "class", "hierarchy-search-spacer svelte-8x1vbd"), u(o, "class", "hierarchy-search-icon svelte-8x1vbd"), u(o, "viewBox", "0 0 20 20"), u(o, "aria-hidden", "true"), u(t, "class", "hierarchy-search-row svelte-8x1vbd"), u(t, "title", k = /*covered*/
      e[94] && !/*selected*/
      e[93] ? "Included by a selected folder" : (
        /*item*/
        e[92].search_display || /*selectedLabelForItem*/
        e[29](
          /*item*/
          e[92]
        )
      )), u(t, "role", "button"), u(t, "tabindex", "0"), u(t, "aria-pressed", w = /*selected*/
      e[93]), K(
        t,
        "hierarchy-search-row-check",
        /*checkboxes*/
        e[28]
      ), K(
        t,
        "hierarchy-search-row-selected",
        /*selected*/
        e[93]
      ), K(
        t,
        "hierarchy-search-row-covered",
        /*covered*/
        e[94] && !/*selected*/
        e[93]
      ), this.first = t;
    },
    m(b, p) {
      q(b, t, p), B(t, r), B(t, n), h && h.m(t, null), B(t, s), B(t, o), M.m(o, null), B(t, a), E.m(t, null), B(t, d), _ || (c = [
        W(t, "click", N),
        W(t, "keydown", m)
      ], _ = !0);
    },
    p(b, p) {
      e = b, /*checkboxes*/
      e[28] ? h ? h.p(e, p) : (h = Nt(e), h.c(), h.m(t, s)) : h && (h.d(1), h = null), C !== (C = F(e)) && (M.d(1), M = C(e), M && (M.c(), M.m(o, null))), S === (S = I(e)) && E ? E.p(e, p) : (E.d(1), E = S(e), E && (E.c(), E.m(t, d))), p[0] & /*searchResults, selectedValue*/
      20480 && k !== (k = /*covered*/
      e[94] && !/*selected*/
      e[93] ? "Included by a selected folder" : (
        /*item*/
        e[92].search_display || /*selectedLabelForItem*/
        e[29](
          /*item*/
          e[92]
        )
      )) && u(t, "title", k), p[0] & /*selectedValue, searchResults*/
      20480 && w !== (w = /*selected*/
      e[93]) && u(t, "aria-pressed", w), p[0] & /*checkboxes*/
      268435456 && K(
        t,
        "hierarchy-search-row-check",
        /*checkboxes*/
        e[28]
      ), p[0] & /*selectedValue, searchResults*/
      20480 && K(
        t,
        "hierarchy-search-row-selected",
        /*selected*/
        e[93]
      ), p[0] & /*searchResults, selectedValue*/
      20480 && K(
        t,
        "hierarchy-search-row-covered",
        /*covered*/
        e[94] && !/*selected*/
        e[93]
      );
    },
    d(b) {
      b && Z(t), h && h.d(), M.d(), E.d(), _ = !1, rt(c);
    }
  };
}
function At(l) {
  let e, t;
  return {
    c() {
      e = A("div"), t = ie(
        /*info*/
        l[5]
      ), u(e, "class", "hierarchy-selector-info svelte-8x1vbd");
    },
    m(r, n) {
      q(r, e, n), B(e, t);
    },
    p(r, n) {
      n[0] & /*info*/
      32 && ve(
        t,
        /*info*/
        r[5]
      );
    },
    d(r) {
      r && Z(e);
    }
  };
}
function ar(l) {
  let e, t, r = (
    /*visible*/
    l[1] && Lt(l)
  );
  return {
    c() {
      r && r.c(), e = tt();
    },
    m(n, s) {
      r && r.m(n, s), q(n, e, s), t = !0;
    },
    p(n, s) {
      /*visible*/
      n[1] ? r ? (r.p(n, s), s[0] & /*visible*/
      2 && se(r, 1)) : (r = Lt(n), r.c(), se(r, 1), r.m(e.parentNode, e)) : r && (lt(), pe(r, 1, 1, () => {
        r = null;
      }), et());
    },
    i(n) {
      t || (se(r), t = !0);
    },
    o(n) {
      pe(r), t = !1;
    },
    d(n) {
      n && Z(e), r && r.d(n);
    }
  };
}
const Ge = 28, Ve = 8, Ke = 6, Ce = 8;
function Ye(l) {
  return Array.isArray(l) ? l.map((e) => String(e)) : l == null || l === "" ? [] : [String(l)];
}
function cr(l, e) {
  const t = {
    folders: (l == null ? void 0 : l.folders) || [],
    items: (l == null ? void 0 : l.items) || []
  };
  return e ? {
    folders: Jt(t.folders),
    items: Ut(t.items)
  } : Wt(t);
}
function Wt(l) {
  return {
    ...l,
    folders: (l.folders || []).map((e) => Wt(e)),
    items: (l.items || []).map((e) => ({ ...e }))
  };
}
function Jt(l) {
  return l.map((e) => ({
    ...e,
    folders: Jt(e.folders || []),
    items: Ut(e.items || [])
  })).sort((e, t) => Ne(e).localeCompare(Ne(t), void 0, { sensitivity: "base" }));
}
function Ut(l) {
  return l.map((e) => ({ ...e })).sort((e, t) => oe(e).localeCompare(oe(t), void 0, { sensitivity: "base" }));
}
function Ne(l) {
  return String(l.name || l.path || "");
}
function oe(l) {
  return String(l.name || l.path || l.value || "");
}
function Ie(l) {
  return String(l.value || l.path || l.name || "");
}
function be(l) {
  return l.value === void 0 || l.value === null ? "" : String(l.value);
}
function hr(l) {
  const e = {}, t = /* @__PURE__ */ new Set();
  function r(s, o) {
    e[s] || (e[s] = []), e[s].push(o);
  }
  function n(s, o) {
    for (const a of s.items || []) r(Ie(a), o);
    for (const a of s.folders || []) {
      const d = be(a);
      d && (r(d, o), t.add(d)), n(a, d ? [...o, d] : o);
    }
  }
  return n(l, []), {
    ancestorsByValue: e,
    folderValues: t
  };
}
function Xt(l, e) {
  return l.search_ancestors.some((t) => e.includes(t));
}
function zt(l, e = "") {
  const t = String(l.path || "");
  if (t) return t;
  const r = Ne(l);
  return e && r ? `${e}/${r}` : r;
}
function Rt(l, e = "") {
  const t = String(l.path || "");
  if (t) return t;
  const r = oe(l);
  return e && r ? `${e}/${r}` : r;
}
function ur(l) {
  return String(l || "").split("/").map((e) => e.trim()).filter(Boolean);
}
function fr(l, e) {
  return l.map((t) => {
    const r = String(t.search_name || oe(t)), n = String(t.search_path || ""), s = String(t.search_text || r), o = String(t.search_display || r);
    return {
      item: t,
      index: s.toLowerCase().indexOf(e),
      name: o,
      path: n
    };
  }).filter((t) => t.index > -1).sort((t, r) => t.index - r.index || t.name.localeCompare(r.name, void 0, { sensitivity: "base" }) || t.path.localeCompare(r.path, void 0, { sensitivity: "base" })).map((t) => t.item);
}
function dr(l) {
  return l.replace(/\\/g, "/").replace(/\.[^/.]+$/, "");
}
function _r(l) {
  return document.body.appendChild(l), {
    destroy() {
      l.remove();
    }
  };
}
function mr(l, e, t) {
  let r, n, s, o, a, d, k, w, _, c, h, F, C, M, I, { elem_id: S = "" } = e, { elem_classes: E = [] } = e, { visible: N = !0 } = e, { value: m = [] } = e, { hierarchy: b = { folders: [], items: [] } } = e, { height: p = 10 } = e, { display_mode: P = "file" } = e, { breadcrumb_separator: z = " " } = e, { sort_hierarchy: f = !0 } = e, { search_empty_label: y = "No matches" } = e, { show_placeholder: L = !1 } = e, { label: Me = "Hierarchy Selector" } = e, { info: it = void 0 } = e, { show_label: st = !0 } = e, { container: Ue = !0 } = e, { scale: Pe = null } = e, { min_width: Te = void 0 } = e, { interactive: te = !0 } = e, { gradio: D } = e, ke, He, fe, ye, Le, we = !1, Fe = !1, de = /* @__PURE__ */ new Set(), X = "", _e = null, Be = null, ot = {}, at = {}, Ae = /* @__PURE__ */ new Set(), ct = "";
  const Yt = `hierarchy-selector-${Math.random().toString(36).slice(2)}`;
  function $t(i) {
    return Re(i);
  }
  function xt(i, g, H) {
    const V = {};
    function G(Y, $ = "") {
      for (const Q of Y.items || []) V[Ie(Q)] = Re(Q, $);
      for (const Q of Y.folders || []) {
        const O = zt(Q, $);
        be(Q) && (V[be(Q)] = ze(O)), G(Q, O);
      }
    }
    return G(i), V;
  }
  function el(i, g) {
    const H = at[i];
    return !!H && H.every((V) => V.some((G) => g.has(G)));
  }
  function tl(i) {
    const g = new Set(i);
    return i.filter((H) => !el(H, g));
  }
  function ll(i) {
    const g = ur(i);
    return g.length ? g.join(z) : String(i || "");
  }
  function ze(i) {
    return s ? ll(i) : i;
  }
  function Re(i, g = "") {
    return ze(Rt(i, g) || String(i.value || ""));
  }
  function rl(i, g = "") {
    return s ? Re(i, g) : oe(i);
  }
  function nl(i, g, H) {
    const V = [];
    function G(Y, $ = "", Q = []) {
      for (const O of Y.items || []) {
        const x = Rt(O, $), ce = x.lastIndexOf("/"), Hl = ce > -1 ? x.slice(0, ce) : $;
        V.push({
          ...O,
          search_key: `i:${Ie(O)}`,
          search_ancestors: Q,
          search_name: oe(O),
          search_path: Hl,
          search_text: rl(O, $),
          search_display: Re(O, $)
        });
      }
      for (const O of Y.folders || []) {
        const x = zt(O, $), ce = be(O);
        ce && V.push({
          ...O,
          search_folder: !0,
          search_key: `f:${x}`,
          search_ancestors: Q,
          search_name: Ne(O),
          search_path: $,
          search_text: s ? ze(x) : Ne(O),
          search_display: ze(x)
        }), G(O, x, ce ? [...Q, ce] : Q);
      }
    }
    return G(i), V;
  }
  function il(i) {
    return ot[i] || dr(i);
  }
  function me(i, g = "change", H = void 0) {
    var V, G, Y;
    t(44, m = Ye(i)), (V = D == null ? void 0 : D.dispatch) == null || V.call(D, "input"), g === "select" && ((G = D == null ? void 0 : D.dispatch) == null || G.call(D, "select", H)), (Y = D == null ? void 0 : D.dispatch) == null || Y.call(D, "change"), Oe().then(ae);
  }
  function ht(i, g = !1) {
    if (!te) return;
    const H = Ie(i);
    r.includes(H) ? me(r.filter((V) => V !== H), "select", { value: H, selected: !1 }) : g || me([...r, H], "select", { value: H, selected: !0 });
  }
  function ut(i, g = !1) {
    if (!te) return;
    const H = be(i);
    r.includes(H) ? me(r.filter((V) => V !== H), "select", { value: H, selected: !1 }) : g || me(tl([...r, H]), "select", { value: H, selected: !0 });
  }
  function Ze(i, g) {
    i.search_folder ? ut(i, g) : ht(i, g);
  }
  function Xe(i) {
    if (!te) return;
    const g = r.filter((H, V) => V !== i);
    me(g, "select", {
      value: r[i],
      selected: !1
    });
  }
  function sl() {
    if (!te || r.length === 0) return;
    const i = r;
    t(10, X = ""), me([], "select", { value: i, selected: !1 });
  }
  function ol(i) {
    de.has(i) ? de.delete(i) : de.add(i), t(9, de = new Set(de)), Oe().then(ae);
  }
  function ae() {
    if (!He || !we) return;
    const i = He.getBoundingClientRect(), g = Math.max(Ge + Ve, Le ? Math.ceil(Le.scrollHeight + Ve) : Ge + Ve), H = h ? g : C, V = i.top - Ce - Ke, G = window.innerHeight - i.bottom - Ce - Ke, Y = V < H && G > V, $ = Math.max(Ge + Ve, Y ? G : V), Q = Math.min(H, $), O = Y ? i.bottom + Ke : Math.max(Ce, i.top - Q - Ke), x = Math.max(Ce, i.left), ce = Math.max(240, Math.min(i.width, window.innerWidth - x - Ce));
    t(24, ct = `top:${Math.round(O)}px;left:${Math.round(x)}px;width:${Math.round(ce)}px;height:${Math.round(Q)}px;`);
  }
  function qe() {
    var g;
    if (!te) return;
    const i = Fe;
    t(8, we = !0), t(56, Fe = !0), i || (g = D == null ? void 0 : D.dispatch) == null || g.call(D, "focus"), Oe().then(() => {
      fe == null || fe.focus(), ae();
    });
  }
  function ft() {
    var i;
    !we && !Fe || (t(8, we = !1), t(56, Fe = !1), t(10, X = ""), (i = D == null ? void 0 : D.dispatch) == null || i.call(D, "blur"));
  }
  function al() {
    qe();
  }
  function cl(i) {
    t(10, X = i.currentTarget.value), qe();
  }
  function hl(i) {
    var g, H, V, G;
    !te || i.target === fe || (H = (g = i.target).closest) != null && H.call(g, "button") || (G = (V = i.target).closest) != null && G.call(V, ".hierarchy-selector-chip") || (i.preventDefault(), qe());
  }
  function dt(i) {
    ke != null && ke.contains(i.target) || ye != null && ye.contains(i.target) || ft();
  }
  function _t(i, g) {
    var H;
    te && (t(22, _e = i), (H = g.dataTransfer) == null || H.setData("text/plain", String(i)), g.dataTransfer && (g.dataTransfer.effectAllowed = "move"));
  }
  function ul() {
    t(22, _e = null), t(23, Be = null);
  }
  function mt(i, g) {
    if (g.preventDefault(), _e === null || _e === i) return;
    const H = [...r], [V] = H.splice(_e, 1);
    H.splice(i, 0, V), t(22, _e = null), t(23, Be = null), me(H);
  }
  function fl(i) {
    i.key === "Escape" ? (i.preventDefault(), X ? t(10, X = "") : ft()) : i.key === "Enter" && k && w.length ? (i.preventDefault(), Ze(w[0], Xt(w[0], r))) : i.key === "ArrowDown" ? (i.preventDefault(), qe()) : i.key === "Backspace" && !X && r.length && Xe(r.length - 1);
  }
  function dl() {
    return Ye(m);
  }
  xl(() => {
    document.addEventListener("pointerdown", dt, !0), window.addEventListener("resize", ae), window.addEventListener("scroll", ae, !0);
  }), $l(() => {
    document.removeEventListener("pointerdown", dt, !0), window.removeEventListener("resize", ae), window.removeEventListener("scroll", ae, !0);
  });
  const _l = (i) => Xe(i), ml = (i, g) => _t(i, g), bl = (i, g) => {
    g.preventDefault(), t(23, Be = i);
  }, gl = () => t(23, Be = null), pl = (i, g) => mt(i, g);
  function vl(i) {
    De[i ? "unshift" : "push"](() => {
      fe = i, t(19, fe);
    });
  }
  function kl() {
    X = this.value, t(10, X);
  }
  function yl(i) {
    De[i ? "unshift" : "push"](() => {
      He = i, t(18, He);
    });
  }
  const wl = (i, g) => Ze(i, g), Fl = (i, g, H) => {
    (H.key === "Enter" || H.key === " ") && (H.preventDefault(), Ze(i, g));
  };
  function Il(i) {
    De[i ? "unshift" : "push"](() => {
      Le = i, t(21, Le);
    });
  }
  function Sl(i) {
    De[i ? "unshift" : "push"](() => {
      ye = i, t(20, ye);
    });
  }
  function Ml(i) {
    De[i ? "unshift" : "push"](() => {
      ke = i, t(17, ke);
    });
  }
  return l.$$set = (i) => {
    "elem_id" in i && t(0, S = i.elem_id), "elem_classes" in i && t(45, E = i.elem_classes), "visible" in i && t(1, N = i.visible), "value" in i && t(44, m = i.value), "hierarchy" in i && t(46, b = i.hierarchy), "height" in i && t(47, p = i.height), "display_mode" in i && t(48, P = i.display_mode), "breadcrumb_separator" in i && t(49, z = i.breadcrumb_separator), "sort_hierarchy" in i && t(50, f = i.sort_hierarchy), "search_empty_label" in i && t(2, y = i.search_empty_label), "show_placeholder" in i && t(3, L = i.show_placeholder), "label" in i && t(4, Me = i.label), "info" in i && t(5, it = i.info), "show_label" in i && t(6, st = i.show_label), "container" in i && t(51, Ue = i.container), "scale" in i && t(52, Pe = i.scale), "min_width" in i && t(53, Te = i.min_width), "interactive" in i && t(7, te = i.interactive), "gradio" in i && t(54, D = i.gradio);
  }, l.$$.update = () => {
    l.$$.dirty[1] & /*value*/
    8192 && t(14, r = Ye(m)), l.$$.dirty[1] & /*hierarchy, sort_hierarchy*/
    557056 && t(13, n = cr(b, f)), l.$$.dirty[1] & /*display_mode*/
    131072 && t(16, s = P === "breadcrumb"), l.$$.dirty[0] & /*normalizedHierarchy, breadcrumbMode*/
    73728 | l.$$.dirty[1] & /*breadcrumb_separator*/
    262144 && (ot = xt(n)), l.$$.dirty[0] & /*normalizedHierarchy, breadcrumbMode*/
    73728 | l.$$.dirty[1] & /*breadcrumb_separator*/
    262144 && t(62, o = nl(n)), l.$$.dirty[0] & /*normalizedHierarchy*/
    8192 && t(11, { ancestorsByValue: at, folderValues: Ae } = hr(n), Ae), l.$$.dirty[0] & /*folderValues*/
    2048 && t(28, a = Ae.size > 0), l.$$.dirty[0] & /*searchQuery*/
    1024 && t(61, d = X.trim().toLowerCase()), l.$$.dirty[1] & /*searchTerm*/
    1073741824 && t(15, k = d.length > 0), l.$$.dirty[0] & /*searchMode*/
    32768 | l.$$.dirty[1] & /*searchTerm*/
    1073741824 | l.$$.dirty[2] & /*flatItems*/
    1 && t(12, w = k ? fr(o, d) : []), l.$$.dirty[0] & /*elem_id*/
    1 && t(27, _ = S ? `${S}-panel` : `${Yt}-panel`), l.$$.dirty[1] & /*height*/
    65536 && t(60, c = Number(p)), l.$$.dirty[1] & /*heightRows*/
    536870912 && t(58, h = c === 0), l.$$.dirty[1] & /*autoPanelHeight, heightRows*/
    671088640 && t(59, F = h ? 0 : Math.max(10, c || 10)), l.$$.dirty[1] & /*panelRows*/
    268435456 && t(57, C = F * Ge + Ve), l.$$.dirty[1] & /*container, focused, elem_classes*/
    34619392 && t(26, M = [
      "hierarchy-selector",
      Ue ? "hierarchy-selector-container" : "",
      Fe ? "hierarchy-selector-focused" : "",
      ...Array.isArray(E) ? E : []
    ].filter(Boolean).join(" ")), l.$$.dirty[1] & /*scale, min_width*/
    6291456 && t(25, I = [
      Pe !== null ? `flex-grow:${Pe};` : "",
      Te !== void 0 ? `min-width:${Te}px;` : ""
    ].join("")), l.$$.dirty[0] & /*open, selectedValue, searchQuery, expanded, normalizedHierarchy, searchResults*/
    30464 | l.$$.dirty[1] & /*autoPanelHeight, panelHeight*/
    201326592 && we && (r || X || de || n || w || h || C) && Oe().then(ae);
  }, [
    S,
    N,
    y,
    L,
    Me,
    it,
    st,
    te,
    we,
    de,
    X,
    Ae,
    w,
    n,
    r,
    k,
    s,
    ke,
    He,
    fe,
    ye,
    Le,
    _e,
    Be,
    ct,
    I,
    M,
    _,
    a,
    $t,
    il,
    ht,
    ut,
    Ze,
    Xe,
    sl,
    ol,
    al,
    cl,
    hl,
    _t,
    ul,
    mt,
    fl,
    m,
    E,
    b,
    p,
    P,
    z,
    f,
    Ue,
    Pe,
    Te,
    D,
    dl,
    Fe,
    C,
    h,
    F,
    c,
    d,
    o,
    _l,
    ml,
    bl,
    gl,
    pl,
    vl,
    kl,
    yl,
    wl,
    Fl,
    Il,
    Sl,
    Ml
  ];
}
class br extends jl {
  constructor(e) {
    super(), Jl(
      this,
      e,
      mr,
      ar,
      Xl,
      {
        elem_id: 0,
        elem_classes: 45,
        visible: 1,
        value: 44,
        hierarchy: 46,
        height: 47,
        display_mode: 48,
        breadcrumb_separator: 49,
        sort_hierarchy: 50,
        search_empty_label: 2,
        show_placeholder: 3,
        label: 4,
        info: 5,
        show_label: 6,
        container: 51,
        scale: 52,
        min_width: 53,
        interactive: 7,
        gradio: 54,
        get_value: 55
      },
      null,
      [-1, -1, -1, -1]
    );
  }
  get elem_id() {
    return this.$$.ctx[0];
  }
  set elem_id(e) {
    this.$$set({ elem_id: e }), j();
  }
  get elem_classes() {
    return this.$$.ctx[45];
  }
  set elem_classes(e) {
    this.$$set({ elem_classes: e }), j();
  }
  get visible() {
    return this.$$.ctx[1];
  }
  set visible(e) {
    this.$$set({ visible: e }), j();
  }
  get value() {
    return this.$$.ctx[44];
  }
  set value(e) {
    this.$$set({ value: e }), j();
  }
  get hierarchy() {
    return this.$$.ctx[46];
  }
  set hierarchy(e) {
    this.$$set({ hierarchy: e }), j();
  }
  get height() {
    return this.$$.ctx[47];
  }
  set height(e) {
    this.$$set({ height: e }), j();
  }
  get display_mode() {
    return this.$$.ctx[48];
  }
  set display_mode(e) {
    this.$$set({ display_mode: e }), j();
  }
  get breadcrumb_separator() {
    return this.$$.ctx[49];
  }
  set breadcrumb_separator(e) {
    this.$$set({ breadcrumb_separator: e }), j();
  }
  get sort_hierarchy() {
    return this.$$.ctx[50];
  }
  set sort_hierarchy(e) {
    this.$$set({ sort_hierarchy: e }), j();
  }
  get search_empty_label() {
    return this.$$.ctx[2];
  }
  set search_empty_label(e) {
    this.$$set({ search_empty_label: e }), j();
  }
  get show_placeholder() {
    return this.$$.ctx[3];
  }
  set show_placeholder(e) {
    this.$$set({ show_placeholder: e }), j();
  }
  get label() {
    return this.$$.ctx[4];
  }
  set label(e) {
    this.$$set({ label: e }), j();
  }
  get info() {
    return this.$$.ctx[5];
  }
  set info(e) {
    this.$$set({ info: e }), j();
  }
  get show_label() {
    return this.$$.ctx[6];
  }
  set show_label(e) {
    this.$$set({ show_label: e }), j();
  }
  get container() {
    return this.$$.ctx[51];
  }
  set container(e) {
    this.$$set({ container: e }), j();
  }
  get scale() {
    return this.$$.ctx[52];
  }
  set scale(e) {
    this.$$set({ scale: e }), j();
  }
  get min_width() {
    return this.$$.ctx[53];
  }
  set min_width(e) {
    this.$$set({ min_width: e }), j();
  }
  get interactive() {
    return this.$$.ctx[7];
  }
  set interactive(e) {
    this.$$set({ interactive: e }), j();
  }
  get gradio() {
    return this.$$.ctx[54];
  }
  set gradio(e) {
    this.$$set({ gradio: e }), j();
  }
  get get_value() {
    return this.$$.ctx[55];
  }
}
export {
  br as default
};

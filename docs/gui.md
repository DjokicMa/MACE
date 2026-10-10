# MACE GUI (`mace gui`)

The MACE GUI shows a folder of calculations in **your own web browser**: its
structures with their symmetry, optimizations step by step, band, DOS, IR,
Raman and FREQ plots, and `materials.db`. You can edit and build structures,
compare them, and set up the next calculation, saving the same JSON settings
file the CLI reads. A small server started by `mace gui` reads the folder; the
page draws everything in the browser, so nothing is installed on your laptop
and the 3D views use your laptop's GPU, even when MACE runs on a cluster node.

The MACE GUI is new in MACE 1.1.5; a few menu items are still to come and say
so ("Not in this version yet").

It writes into the served folder only what you save there (a structure, a
CIF, a settings JSON, Calculation Setup's files and the decks you ask it to
write) and, when you ask, a `materials.db` built from the folder's outputs. It
never writes over a calculation's outputs, and keeps the previous version of
every file it replaces; [What the GUI writes into your
folder](#what-the-gui-writes-into-your-folder) lists everything, and where.

**Start here:** [Run it locally](#run-it-locally),
[First run from your computer](#first-run-from-your-computer) (MACE on a cluster),
[Troubleshooting](#troubleshooting).

## The window

The page is laid out like a desktop structure editor:

- a **menu bar** (File, Edit, View, Select, Build, Tools, Calculation,
  Analysis, Help) with a command search (**F1** or **Ctrl+Alt+K**; ⌥⌘K on a Mac), the
  theme switch and the settings;
- a **toolbar** with the tools, the view toggles (cell, bonds, labels,
  boundary atoms, axes, projection), view along a/b/c, the display range,
  the P1/Sym switch, Clean and Calculation Setup;
- the **Explorer** on the left (open documents, the folder grouped into
  outputs, results, structures and this folder; long names lose their middle,
  and the tooltip shows the full path);
- **document tabs** in the middle: each file you open is a tab. A `.cif`,
  `.d12` or single-point `.out` opens as a structure, an OPT `.out` as a
  trajectory, a BAND/DOSS file as a plot and `materials.db` as a database;
- the **Inspector** on the right (selection, structure, cell, symmetry and
  display for a structure; the selected material for a database), which the
  **Calculation Setup** panel (the old Deck tab) replaces while it is open.
  In a window narrower than 1100 px it is a drawer over the viewport, shut
  until you ask for it (View › Inspector, **Ctrl+Alt+B**);
- a **bottom panel** with Atoms, Measurements, History, Timeline (the OPT
  energy chart and the cell, bond and displacement tables), Tasks and
  Problems;
- a **status bar**: the connection to the server, the selection, the space
  group, the cell parameters, the atom under the pointer and whether the
  document is modified.

Every menu item, toolbar button, tooltip and shortcut comes from one
command list, so they always agree; **?** shows every shortcut and
**F1** (or Ctrl+Alt+K) finds any command by name. Items that are not built yet are shown
disabled, with "Coming soon". Single-key shortcuts (`1`, `V`, `?`) never fire
while you type in a field. **Ctrl+,** opens the settings; they are kept in
`~/.config/mace/gui.json` (or `$XDG_CONFIG_HOME/mace/gui.json`), outside the
served folder, with the rest of your profile (see
[Settings and your profile](#settings-and-your-profile)).

**Zen Mode** (View › Zen Mode, **Ctrl+Alt+Z**) hides the menu bar, toolbar,
activity bar and status bar. To leave it, click **Exit Zen Mode** at the top
right (it fades after a few seconds and comes back when you move the pointer
to the top), press **Esc** (when no dialog, menu, field, tool or selection
wants it first), use the viewport's right-click menu, or press Ctrl+Alt+Z
again. A reload keeps Zen Mode, with the button shown.

The browser tab reads "*document* — *folder* — MACE"; the menu bar shows the
folder's path, shortened: `~` for your home folder, `$SCRATCH/…`, `$WORK/…` or
`$PSCRATCH/…` for a folder under those (when your cluster sets them), with
the full path as its tooltip.

The page stays light over an ssh tunnel: it starts with one request, the 3D
viewer loads after the first paint and plotly only when a plot opens, every
asset is cached under a versioned URL (a reload sends about 8 KB), and a
heartbeat over Server-Sent Events drives the status bar's **Connected**. If
the tunnel drops, a banner shows the `ssh -L` line to run again. The 3D
views paint only when something changes, at most once per frame, and the
server's caches stay under `MACE_GUI_CACHE_MB` (512 MB by default).

## Keys, the command search, settings and context menus

### Keyboard shortcuts and how to change them

Every shortcut belongs to a command, and the menus, toolbar tooltips,
context menus, the command search and the shortcut sheet all read the keys
from that one list. **?** (or **Ctrl+/**) shows the sheet in four columns:
Global, File and Edit, Select and Tools, and View. Its **Print** button
prints it as one landscape page.

To change a key, open **Settings › Keyboard** (or **Edit keybindings…** on
the sheet). Click a binding and press the new keys; **+** adds a second
key. The change applies at once, everywhere the key is shown.

- **Esc** cancels the recording and **Backspace** removes the binding. To
  bind Esc or Backspace themselves, use **Enter as text…** and type `Esc` or
  `Backspace` (any chord, such as `Mod+Shift+K`, can be typed there).
- If another command already uses the key, the editor names it and asks
  before it takes the key over ("Ctrl+D is also Duplicate. Replace it?").
  **Pick another key** keeps recording.
- Enter and Space on the recorder's buttons press them. To bind Enter or
  Space, press it with the recorder panel itself focused, or type it.
- **Reset** on a row, **Reset all keybindings**, and **Export / Import
  keymap.json** are on the same page. Only your changes are saved: a list of
  `{"command": id, "keys": [...], "remove": true?}` entries, in Settings'
  `keymap`.

`Mod` means Ctrl on Linux and Windows and ⌘ on a Mac. Keys are matched by
their position on the keyboard (`event.code`), so they work with any
layout.

**Keys the browser owns.** These are refused, with the reason, because a
page cannot reliably use them: Ctrl+N, Ctrl+T, Ctrl+W, Ctrl+Shift+N/T/W,
Ctrl+Tab, Ctrl+L, Ctrl+Q, Ctrl+Shift+P, Ctrl+PageUp/PageDown, Ctrl+1..9, F5,
F11 and F12. That is why New Structure is **Alt+N**, Close Tab **Alt+W**,
Set Coordinates **Ctrl+Shift+U** (Ctrl+Shift+T reopens a closed browser tab)
and Focus Viewport **Ctrl+Alt+1** (Ctrl+1 is browser tab 1).

**Keys the editor warns about** (you can still use them):

- developer-tools keys that some browsers take first: Ctrl+Shift+I, J, C, K,
  E and M;
- keys that leave the page or switch tabs: Alt+Left and Alt+Right (Back,
  Forward), Alt+Home, Alt+1..9 (tab switching on Linux) and Backspace (Back
  in some browsers). While a command is bound to one of these, MACE keeps
  the key from the browser even when the command cannot run (it says why in
  a toast). So Alt+Left with the viewport unfocused does not go Back.
  Unbound keys stay the browser's.

Leaving or reloading the page with a modified document asks first.

**Menus from the keyboard.** Alt+F, Alt+V, Alt+S, Alt+T, Alt+C, Alt+A and
Alt+H open File, View, Select, Tools, Calculation, Analysis and Help. Edit
and Build have no Alt key by default: Firefox takes Alt+E and Alt+B for its
own Edit and Bookmarks menus.

**Keys Firefox keeps.** Measured with real key presses in a visible Firefox
(`tests/gui/test_browser_keys_xvfb.py`), Firefox acts on Alt+E and Alt+B
(its menus), Ctrl+K (web search), Ctrl+Shift+J (browser console),
Ctrl+Shift+M (responsive design mode), Ctrl+Shift+G (find previous) and
Ctrl+Shift+A (add-ons) before the page sees them, so MACE doesn't use them:
the command search is F1 or Ctrl+Alt+K, Select by Element Ctrl+Alt+E, Select
Bonded Neighbors Ctrl+Alt+N, Save Settings JSON Ctrl+Alt+J, Clear
Measurements Ctrl+Alt+M, Clean Geometry Ctrl+Alt+G and Select None Esc or
Ctrl+Alt+A. The keymap editor refuses those keys with the reason. F12 always
opens the developer tools. **F10** focuses the menu
bar, and the arrow keys then reach every menu. The mnemonics are listed
under Settings › Keyboard › Show: Menu mnemonics, and can be moved or
removed like any key.

**Viewer and timeline keys:** the arrow keys
turn the view by 15° (Shift: 1°) and **+**/**−** zoom while the viewport has
focus; **,** and **.** step through an optimization (Shift: first and last
step), **Space** plays it, and **Ctrl+Alt+O** switches overlay and side by
side; **Ctrl+G** jumps to an atom; **Shift+F10** opens the context menu of
the focused row or tab. If a browser takes one of these keys despite MACE,
each has a `Mod+Alt+…` alternative (the atom styles: `Ctrl+Alt+1..4`).

### The command search (F1, Ctrl+Alt+K)

Type any part of a command's name; the first character picks a mode:

| Prefix | Searches | Also |
|---|---|---|
| (none) or `>` | commands, recently used first | **Enter** runs, **Ctrl+Enter** runs and keeps the search open |
| `#` | files in the folder | **Ctrl+P**, File › Open… |
| `%` | open documents | **Ctrl+E** |
| `@` | atoms of the open structure (`@O5`) | **Ctrl+G**; centers the view on the atom |
| `:` | a step of the open optimization (`:12`) | |
| `=` | a calculator: `= 0.5 0 0.5` gives Cartesian Å in the open cell, `= c 2.1 0 2.1` gives fractional | **Enter** copies the result |

Every command is listed, including the ones not built yet (with "Coming
soon"); Settings › Advanced › Show command ids adds each command's id.

### Settings (Ctrl+,)

Eight sections: Appearance, Viewport, Units and numbers, Editor, Keyboard,
Files and data, MACE defaults and Advanced, with a search over all of them.
Changes apply at once; a **Reset** link shows next to anything you changed.
Rows marked "Not used in this version yet" are kept for a later version.

- **Viewport** settings redraw the 3D views live with the camera kept: color
  scheme (VESTA, the default, or Jmol), atom scale, bond radius, bond
  tolerance (drawn bonds only; the bond table keeps 1.2), bonds between
  metals, labels and their size, background, cell color, the axis triad,
  the cell and the boundary atoms.
- **MACE defaults** show what the CLI does when you leave a value blank
  (for example "CLI default: HSE06"), read from the CLI's own code. A blank
  GUI value means "use the CLI default".
- **Advanced** exports, imports and resets the settings (the file is a
  browser download; nothing is written into the served folder).

The server checks each value's type and range when it saves; a bad value is
named in a toast and goes back to its previous value, while the others are
saved.

### Settings and your profile

Everything you choose in the GUI is kept on the machine that runs the
server, in `~/.config/mace/gui.json` (or `$XDG_CONFIG_HOME/mace/gui.json`),
outside the served folder: the Settings you changed, the layout (panel
widths, the bottom tab, collapsed Inspector sections, the Explorer's
sections, database column widths, Setup's sections, each folder's split into
two editor groups), remembered choices (the
last Export Image choices, the New Structure form, the Clean method, the
Submit dialog's last options, the last commands of the command search, recent
elements), the Recent files of each folder and the Recent folders of File ›
Open Folder. Each document's own view options (its display range, boundary atoms,
rendering) are kept next to it, in `gui-docprefs.json`.

So everything comes back when the local port changes (a new ssh tunnel, VS
Code's port forwarding), in another browser, and on another computer that
shares your home folder, except:

| Kept in this browser only | Why |
|---|---|
| Unsaved drafts of edited structures and of Calculation Setup | they are work in progress, offered again after a reload in the same browser |
| Zen Mode and the narrow-window Inspector drawer | a small screen should not open another computer in Zen |
| The "overwrite this file?" question, asked once per file | a safety prompt |
| A plot's own options and its undo history, Run Analysis's and the energy card's options, Follow the run | per browser for now |

A change made just before you close the tab is still saved (it is sent as
the page closes). The browser keeps a copy (localStorage) so the page starts
at once; the server's copy wins when they differ, unless a change made here
has not reached the server yet. The page's theme, density and UI scale come
with the page itself, so a new port does not flash the defaults. Two servers
of your account (two folders, or two cluster nodes sharing your home) can
save at the same time: each change is merged per key under a file lock, and
the file is read back to catch a lock that did not work across nodes. The
first save of this version keeps a copy of the older file as
`gui.json.v1.bak`.

Settings › Advanced:

- **Export settings…** downloads your profile as `mace-gui-profile-<date>.json`.
  Choose what goes in: Settings, Layout and panels, Remembered choices (on by
  default), and Recent files and folders and Per-document view options (off
  by default: they hold paths on this computer; leave them off when sharing
  the file).
- **Import settings…** reads such a file (or a settings file from an earlier
  version), says what it will replace, and merges it after you confirm; a
  value the server refuses is named and the others are applied.
- **Reset all…** asks which parts to reset (Settings and keybindings, Layout
  and Remembered choices by default) and first saves a copy of your profile
  next to it (`gui.json.before-reset-<time>.json`; the newest 3 are kept).

These cover the GUI profile only; the profiles of `mace gui connect` are
never exported, imported or reset here.

### Context menus

Right-click (or press the Menu key or Shift+F10):

- **an atom**: Select Equivalents, Select Element, Select Neighbors, Replace
  Element…, Set Coordinates…, Delete, Measure from Here, Center View Here,
  Copy Coordinates (Fractional / Cartesian, of the image you clicked), Properties;
- **empty space** in the viewport: Add Atom Here, Paste, Select All, View
  Along, Reset View, Display Range, Export Image…;
- **a file** in the Explorer: Open, Open to the Side, Open as Text (a
  read-only text tab), Compare With Active, Plot, Copy Path, Reveal in
  Folder, Set Up Calculation;
- **a tab**: Close, Close Others, Close to the Right, Copy Path, Reveal in
  Explorer.

Each entry is a command, so it shows its key and can be run from the
command search too. Esc closes the menu and puts focus back on the file,
tab or view it was opened for.

### How much of the machine it uses

`mace gui` shares the machine politely (cluster dev nodes are shared, and a
workstation may run other jobs on its GPUs):

| Resource | Default | Setting | Environment variable |
|---|---|---|---|
| CPU workers (the process pool and the server's own symmetry threads, counted together) | a quarter of the usable cores, 1 to 8, within SLURM's allocation; nice 10 | Advanced › CPU workers (`max_workers`, 0 = auto) | `MACE_GUI_MAX_WORKERS` |
| Cache memory | 512 MB, at most a quarter of RAM, at least 64 MB | Advanced › Cache limit (`cache_mb`) | `MACE_GUI_CACHE_MB` |
| GPU for MACE's own GPU features (the server uses none today) | GPU 0; None keeps everything on the CPU | Advanced › GPU (`gpu_device`) | `MACE_GUI_GPU` |

**The process pool and the Tasks panel.** Builds over 2000 atoms, Clean,
the symmetry sweep, defect searches and the explorer's file details run in
worker processes: started with the first such job, at most the CPU-workers
number of them, at nice 10 with one BLAS thread each, and stopped after 60 s
idle (about 0.3 s to start one again, about 100 MB each). They share one CPU
budget with the server's own symmetry threads, so the two never add up to
more than the cap. A job answers at once when it takes under 0.3 s;
otherwise it becomes a task: the dialog shows its progress, and closing the
dialog hands it to the **Tasks** panel (bottom), where it can be canceled
(it stops within 2.5 s), retried, or its error copied. Opened files, built
structures that are too big to edit and the server's cache share the one
cache-memory budget above.

An environment variable wins over the setting. A setting saved on a bigger
host (say `max_workers: 64` on a cluster node) is kept as it is and limited
to what the current host has when it is used; Settings › Advanced shows
both numbers.

**Which GPU the browser draws on.** The browser, not the server, does the
3D drawing, and the operating system picks its GPU: by default GPU 0, the
one that usually drives the display. On a Linux workstation with a second
NVIDIA GPU, `mace gui --open --gpu 1` (for example) starts the browser with
PRIME render offload on GPU 1 (GPU *n* ≥ 1 is the X provider
`NVIDIA-G`*n−1*, see `xrandr --listproviders`). It only affects a browser
that `mace gui` starts; a browser that is already running keeps its GPU. On a shared node, do not raise the CPU
workers above the default.

The headless browser used by the tests and benchmarks always draws in
software (Mesa's llvmpipe) and touches no GPU.

## What it shows

| Tab | What you get | MACE code it uses |
|---|---|---|
| **Structure** | A 3D view of a `.cif`, a `.d12` or a CRYSTAL `.out`. You get the unit cell box, the atoms on the cell faces (as VESTA draws them), a 2×2×2 supercell, bonds, atom labels, and VESTA element colors (Jmol colors as an option). The camera fits the cell and every atom, also in narrow panes. A SLAB's box spans the slab itself; **Vacuum** pads it with 4 Å on each side. Hovering over an atom shows its fractional coordinates and Wyckoff letter. Alongside the view are the reduced formula with Z, the space group (spglib), the lattice parameters and an atom table. | ase (CIF), `d12_parsers.CrystalInputParser`, `seekpath_interface._geometry_blocks` |
| **Editor** | Edit a 3D structure, or build a new cell from lattice parameters (optionally a space group plus the asymmetric unit). See [Editing](#editing) below. | ase, spglib |
| **Compare** | A VESTA-style before and after, for an OPT `.out` (frame 0 against the selected frame), for an output against its input (**Analysis › Compare with Input**), for any two structures (**Tools › Compare Two Structures**), and for Clean. Four looks, cycled by Ctrl+Alt+O: overlay (the initial as a ghost), two-tone, side by side (linked cameras), and a displacement map colored by \|Δr\|. Tables of Δcell, Δbond, displacements and Δsymmetry in the Timeline panel. See [Compare](#compare). | `_geometry_blocks`, spglib, scipy (optional) |
| **Plots** | Band structures, DOS, IR and Raman spectra as documents with their own options, and FREQ normal modes animated in the viewport. See [Plots](#plots-band-dos-ir-raman-and-freq-modes). | `Plotting/ipBANDS_V2.py`, `Plotting/ipDOS_V2.py`, `spectra_api.read_spectrum`, `vibmode_viewer.Crystal23FreqParser` |
| **Database** | A read-only document for a `materials.db`: sorted, filtered and paged on the server, so 100,000 materials scroll like ten. See [The Database document](#the-database-document). | the MACE schema, through a read-only sqlite connection |
| **Calculation Setup** | The CLI's own questions as a form, the deck the printed command writes, and Run Preflight. See [Calculation Setup](#calculation-setup). | `mace convert`, `mace opt2d12`, `mace opt2d3`, `mace preflight` |

### The GUI → JSON → CLI loop

Calculation Setup has no deck writer of its own. Every value it shows comes
from the CLI: its defaults, its menus and its questions. Previewing runs the
printed command in a scratch mirror of the folder and shows the deck the CLI
writes. **Save Settings JSON** then writes that same JSON, so the printed
command (for example

```
mace convert --batch --options_file gui_options.json --cif_dir structures --output_dir decks/
```

) writes the deck you previewed, byte for byte. `tests/gui/test_setup_routes.py`
checks this against the manual, interactive route of each command for the
matrix of calculation types.

### Formulas

The formula is the **reduced formula** with the elements in ascending Pauling
electronegativity, as pymatgen writes it: MgO, LiF, SiO2, GaAs, BaTiO3, CsPbI3,
NaHCO3. Z, the number of formula units in the cell, is shown beside it
(conventional MgO: Z = 4; diamond: C, Z = 8). `materials.db` keeps the cell's
full formula in Hill order.

## Editing

A CIF, a 3D `.d12` or the structure of an `.out` opens as a structure
document, and it is edited where it is shown: there is no separate editor
mode. The editor's code loads with the first structure document, so an idle
page or a reload fetches none of it. Editing needs a 3D periodic cell of at
most 5000 atoms; slabs, larger cells and runs say why their editing commands
are off (a run's step can be edited with **Edit This Step**, below).

### The pointer

The pointer works the same way in every tool, as in OVITO, Avogadro and
Blender:

- the wheel zooms toward the cursor, a middle-drag pans, a right-drag orbits,
  and Alt+left-drag orbits from any tool (for trackpads);
- a right-click without dragging opens the atom menu (on an atom) or the
  viewport menu (on empty space);
- the left button belongs to the tool. With **Select** (V) a click picks an
  atom, Shift+click adds, Ctrl+click toggles, a drag draws a box, a
  double-click selects the symmetry orbit and a triple-click the bonded
  fragment. **Lasso** (L) draws a free polygon; **Orbit** (O), **Pan** (H) and
  **Zoom** (Z) turn the left button into that motion.

Selected atoms get an amber ring and a translucent shell, and the atom under
the pointer a blue ring (the halo). The rings are drawn over the 3D view
after every frame, so they stay sharp and follow the camera; they are left
out of exported images. The status bar shows the atom
under the pointer with its Wyckoff letter and coordinates (Shift for
Cartesian).

Large cells are drawn at a lower level of detail, and a badge in the corner
says so: up to 2000 displayed atoms everything is drawn (LOD 1); up to
20,000 bonds become lines and labels show on hover only (LOD 2); up to
200,000 bonds are left out unless you turn them on (LOD 3); beyond that the
drawing is simplified (LOD 4). Settings › Viewport › Quality halves (Low) or
doubles (High) these limits. Bonds are found in a web worker
on a cell-list grid (10,648 atoms in a few tens of milliseconds), and box and
lasso selection run there too.

### Tools and keys

- **Move** (G): with a selection, G makes it follow the pointer in the
  screen plane, Blender style; a drag on a selected atom does the same. While
  moving, X, Y or Z locks a Cartesian axis and Shift+X/Y/Z a cell axis, typed
  digits give the offset in Å, Ctrl snaps to 0.05 of a cell axis, Enter or a
  click applies, and Esc or a right-click cancels. These keys only mean this
  while a move is on; the shortcut sheet lists them as their own layer.
- **Add Atom** (A): a click adds the current element on the plane through the
  cell's center facing you; Shift+click on an atom adds one bonded to it, away
  from its neighbors. E (or the element field in the tool chip) picks the
  element; while that field has focus, letters are typed there, not run as
  shortcuts. The viewport menu's **Add Atom Here** adds one where you clicked.
- **Replace** (E): a click replaces the atom; a drag paints, and the whole
  stroke is one undo step. With a selection, E opens the periodic table.
- **Measure** (M): click 2, 3 or 4 atoms for a distance, an angle or a
  dihedral; Shift+click chains distances. Ctrl+M measures the selection.
  Measurements are listed in the Measurements panel (copyable as TSV) and
  follow the atoms through edits.
- **Bond** (B): click two atoms to add or remove a bond from the drawing only.
- **Delete** (Delete, Backspace), **Set Coordinates…** (Ctrl+Shift+U, with
  expressions such as `1/3`), **Edit Cell…** (Ctrl+Shift+L, with a dashed
  preview of the new cell), the Alt+arrow nudges (Shift: ten times the step),
  the selection commands (Ctrl+Alt+E by element, Alt+Q equivalents, Alt+R within a
  distance, Ctrl+Alt+N bonded neighbors, Ctrl+] / Ctrl+[ grow and shrink, Alt+I by
  index such as `1-4, 9, 12-`, by fractional coordinate such as `z > 0.5`,
  and named selection sets).
- The views: 1, 2, 3 look along a, b, c and Shift+1/2/3 along a\*, b\*, c\*;
  Alt+U along any [u v w]; 5 switches orthographic and perspective; F frames
  the selection. The axis triad in the corner shows the cell axes.
- The **Atoms** panel lists every atom (only the rows in view are in the
  page, so 8000 atoms scroll smoothly), follows the selection both ways,
  edits a cell in place (Enter, Tab) and takes a pasted column of numbers as
  one Set Coordinates step. The **History** panel (Ctrl+Shift+H) lists the
  undo steps from "Opened MgO.cif"; a click goes back to any of them, and the
  later rows stay gray and redoable until the next edit.

### Sym and P1

Every document is in one of two modes, shown in the status bar and the
inspector:

- **Sym** (the default for a structure with more than one symmetry
  operation): the space group's operations are kept, and an edit acts on the
  whole orbit. Replacing Mg in MgO replaces all four Mg, deleting an atom
  deletes its equivalents, and moving one moves every image. A move off a
  special position offers **Keep** (the move projected onto what the site
  allows) or **Allow** (the orbit grows and the composition changes).
- **P1**: every atom on its own. The status bar still shows the space group
  of the structure as it is, in amber with "↓ from Fm-3m" when it dropped
  below the group the document had.

Switching from P1 back to Sym on a lowered structure asks whether to
**Symmetrize** first (spglib, one undo step) or lock the lower group. The
live space group during a drag comes from `/api/symmetry`: one request at a
time, 150 ms after the last change, and spglib runs on at most the workers
the [resource limits](#how-much-of-the-machine-it-uses) allow.

### Saving

- **Save** (Ctrl+S) writes a CIF in place, with every atom (P1 CIF). The
  first time for each file it asks "Overwrite MgO.cif? The original will be
  kept as MgO.cif.bak", with **Don't ask again for this file**
  (Settings › Editor › Confirm before overwriting turns the question off).
- The backup is `MgO.cif.bak`, next to the file. It is made only if there is
  none, and never replaced, so it is always the oldest original MACE saw.
  Settings › Editor › Keep a .bak turns it off.
- If the file changed on disk since it was opened (another program, `git
  checkout`), Save says so and offers **Overwrite**, **Save As…** or
  **Cancel**. This check runs even with the confirmation turned off.
- A `.out`, a `.d12`, a step of a run and a new structure always go through
  **Save As…** (Ctrl+Shift+S), which suggests `<name>_edited.cif`. It picks a
  folder inside the served directory (nowhere else) and a format: CIF with
  every atom, a symmetrized CIF (spglib's standard cell), POSCAR or extended
  XYZ (Settings › Files and data › Save format sets the first choice).
- **Save All** (Ctrl+Alt+S) saves every modified document and stops at the
  first that needs Save As. **Revert File** asks, then reopens the file and
  drops the unsaved changes and the undo history.
- The classic panel's **Save CIF** button opens the same Save As, with its
  name and CIF kind filled in.

A modified document has a dot on its tab, "● " in the window title,
"Modified" in the status bar and an accent on the Save button. Closing it or
the page asks first.

### Drafts

While a document is modified, its structure is kept as a draft in the
browser's local storage every Settings › Editor › Autosave drafts seconds
(10 by default; 0 turns it off), and at once when the page is hidden or
closed or the connection to the server is lost. Nothing is written into the
served directory. Opening the page again offers "Restore unsaved changes to
MgO.cif from 14:02? **Restore** / **Discard**". Drafts over 2 MB are skipped
with a note.

The browser keeps drafts per origin, that is per `http://127.0.0.1:<port>`:
a draft written under one port (or, over a tunnel, one local forwarded port)
is not offered under another. `mace gui` uses the same port every time
unless `--port` says otherwise, and it prints a reminder when it does.

### When the server goes away

- **The server restarted** (Ctrl+C and start again, a new token): every
  request is refused, so the page stops sending, keeps your edits in the tab,
  saves the drafts at once and shows "Lost the connection to the MACE server".
  Reopen the URL the server printed: the drafts are offered there. A fixed
  token (`MACE_GUI_TOKEN`, or `--token`) keeps the same URL across restarts,
  so the page carries on instead.
- **The server let go of a document's history** to stay within its memory
  budget: the page rebuilds the session from the structure it shows and
  says "This document's undo history was released to save memory"; the
  structure is kept and the earlier undo steps are gone.

### Memory

The undo history, the open sessions and the cache of page files share one
budget, Settings › Advanced › Cache limit (`cache_mb`, 512 MB by default, or
`MACE_GUI_CACHE_MB`): the history may use at most half of it and the page
files a quarter. Each document keeps up to Settings › Editor › Undo steps
(200) steps; one step of a 5000-atom structure takes about 125 KB. When the
budget is full the oldest undo steps of the least recently used documents go
first. Closing a tab lets the server drop its session at once.

### Export Image and New Structure

- **Export Image…** (Ctrl+Shift+X) saves a PNG of the view at the viewport's
  size, 2×, 4×, 1920×1080, 3000×3000 or a custom size (at most 8192 pixels a
  side, less if the graphics can't draw that much), with a transparent
  background, the axis triad, the measurement labels and the cell as options.
  The browser downloads it as `<name>_<view>.png` (`MgO_a.png` after viewing
  along a). **Copy Image to Clipboard** (Ctrl+Shift+C) copies the view, or
  says why the browser refused and offers Export Image.
  **Physical size** sets a printed width (inches or cm) at a resolution (72,
  150, 300, 600 or a custom DPI, Settings › Export resolution, 150 by default):
  3.5 in at 300 DPI is 1050 pixels wide. Every exported PNG says its DPI (a
  pHYs chunk), so it keeps its printed size in a document. SVG export has
  **Shading**: Flat (plain colors, the default) or Shaded (one radial gradient
  per element); fog and ambient occlusion are raster only. Plots have the same
  dialog under **Image…** (their PNG and SVG buttons still download at once).
- **New Structure…** (Alt+N) builds a cell from a b c α β γ, a space group by
  number or Hermann-Mauguin symbol (`225` or `F m -3 m`) with its setting,
  and the asymmetric unit, one "El x y z" line per site. It opens as an
  untitled document; Save goes to Save As.
- **Edit This Step** (in the Timeline panel and the command search) opens the
  step of a run on screen as a structure document.

### The classic panel

The first editor is still in the inspector under "More editing (classic)"
([below](#the-classic-editor-panel)): typed coordinates, supercell,
primitive cell, bond display limits, Clean and Preview deck. It acts on the
active document through the same requests, history and save path. The Build
menu's dialogs (supercell, slab, defects, transform, Clean) do the same with
more options.

## The classic editor panel

Open a structure with **Edit this structure** on the Structure tab (3D
crystals only), or expand **New cell from lattice parameters**. Every edit
runs on a session on the server, which keeps the full undo/redo history
(**Undo**/**Redo**, Ctrl+Z/Ctrl+Y).

- **Select** atoms by clicking them; Shift+click adds to the selection, Esc
  clears it.
- **Add** an atom at typed fractional or Cartesian coordinates (**From
  selection** fills in the center of the selected atoms). Type the element;
  the field suggests every symbol.
- **Delete**, **Replace** (with the typed element) and **Move** the selection.
  The arrow keys nudge it along x and y and PgUp/PgDn along z, by the step you
  set. They do nothing while a text field has focus.
- **Cell**: change a, b, c, α, β, γ, with or without scaling the atoms; make a
  supercell; reduce to the primitive cell; **Symmetrize** (spglib refine at the
  tolerance you set). The space group is recomputed after every edit.
- **Bond display** limits a pair's bonds to a length you choose. This changes
  the drawing only.
- **Symmetry-aware** editing captures the space group operations when you
  switch it on. An edit then acts on the whole orbit: adding an atom adds all
  its symmetry images, and deleting or replacing an atom does the same to all
  its equivalents. Moving an atom moves each equivalent atom by the image of the
  displacement. If a move takes an atom off a special position, its orbit grows
  (4a → 32f in Fm-3m) and the composition changes, so the editor asks first.
  Supercell, primitive, symmetrize, a cell change and Clean can change the
  operations (stretching cubic MgO to a ≠ b ≠ c leaves Fmmm), so the group is
  captured again afterwards, and the status line says so.
- **Save CIF** opens Save As (above) with the name and the CIF kind (a
  symmetrized CIF: spglib's refined cell, space group and asymmetric unit; or a
  P1 CIF with every atom) filled in, in the document's folder. It then selects
  that file in the Deck tab. **Preview deck** runs cif2d12 on the edited
  structure with the Deck tab's JSON, using the CIF name you typed, so the
  preview is the deck the saved CIF gives. A P1 CIF gives a P1 deck unless the
  JSON's `symmetry_handling` is SPGLIB.

`tests/test_gui_editor.py` saves symmetrized CIFs of MgO (225), Si (227), hBN
(194), the R-3c Ag2Br3 cell (hexagonal setting) and a P21/c cell, writes their
decks with cif2d12, expands each deck again, and checks the space group,
volume and neighbor shells against the edited structure.

### Clean geometry

**Clean** is a rough pre-relaxation for tidying hand-placed atoms. It is not
physics, and its result is a starting geometry for CRYSTAL, not a relaxed
structure. It is a periodic spring model in ase (`mace/gui/clean.py`):

- pairs closer than 1.2 × the sum of their covalent radii are bonded (the
  same rule the viewer draws by; no metal-metal bonds). Each bond relaxes toward
  the sum of the covalent radii;
- other pairs closer than 1.8 × the sum of the radii repel softly. Bond springs
  alone leave zero-energy motions (6 in conventional Si, 9 in conventional MgO
  or NaCl, by the rank of the rigidity matrix), and the repulsion removes them;
- FIRE moves the atoms; **keep the space group** adds ase's `FixSymmetry`.
  **Relax the cell** is off by default: covalent radii underestimate ionic
  bonds (Na-Cl 2.68 Å against 2.82 Å measured), so an ionic cell shrinks by
  about 5 %. The cell responds to the bond springs only.

**Build › Clean Geometry…** (Ctrl+Alt+G, and the classic panel's **Clean**)
opens a panel that runs Clean in the process pool and shows a **preview**:
the cleaned structure as a ghost, an arrow per moved atom, and a summary
(maximum and RMS shift, the space group after). Nothing changes until
**Accept**, which is one History entry "Clean Geometry"; **Discard** (or Esc,
or Ctrl+Z while the preview shows) adds nothing. While a preview shows, the
document can't be edited (the status bar says why); switching tabs keeps the
proposal. The time limit stops it with a "not converged" note, and Cancel
stops a long one within 2.5 s. An all-metal cell or a lone atom has no bonds, and
Clean says there is nothing to clean. The tests check that distorted MgO, Si
and NaCl cells return to their bond lengths, keep their space group with
FixSymmetry, and that an ideal cell does not move.

**Method** picks how Clean moves the atoms: **Springs** (the default, the model
above, with the live preview), **GFN-FF** (the external `xtb` program, 6.7 or
newer), an **ML potential** (MACE-MPA-0 from ACEsuit's mace-torch, a
third-party model unrelated to this MACE, or CHGNet 0.3.0, run in their own
Python environment) or **Auto**. A method that is not installed is grayed out
with the commands that set it up; **Recheck** looks again, and the GUI itself
never installs or downloads anything. GFN-FF, ML and Auto take a minute or
more, so they do not preview on every change: **Run** starts one (a second Run
replaces it; closing the dialog cancels it), and Run Longer goes up to 600 s.
The dialog also has the **Charge** (prefilled from the document; required for a
document opened from a CHARGED/CHEMOD/DOPING deck), **Symmetrize first** (off;
it can raise the space group, which CRYSTAL then keeps) and, under Advanced,
fmax and steps. The readout says which method ran, why another was refused or
rejected (the spring result is then kept), that the result is a starting
geometry for a CRYSTAL OPT, and offers **Copy citation**. A document from a
deck whose records edit its geometry (SUPERCEL, ATOMREMO, ...) is the host
geometry: Clean offers the springs only and says so. The same pipeline is the
CLI's `mace preopt`; see [preopt.md](preopt.md) for the methods, their limits,
licenses and citations.
Any of them would be an optional ase calculator.

## Building structures

The **Build** menu (and the Inspector's Cell section, the atom menu, and the
classic panel's Make/Primitive/Symmetrize buttons) opens a dialog with a live
preview in the viewport. Each dialog has a **Result** control: **New tab**
opens the result as a new, unsaved document named for what it is
(`MgO_2x2x2`, `MgO_001_5L`, `MgO_2x2x2_V_O_4b`) that remembers where it came
from; **This structure** applies it to the open document as one History entry.
The choice is remembered per dialog. A slab is always a new tab (it never
replaces its bulk).

- **Supercell…** (Ctrl+Alt+P): n1 × n2 × n3, or a 3 × 3 integer matrix with
  its determinant (entries within ±30), and presets (2×2×2, √2×√2×1 R45°, and
  √3×√3 R30° for hexagonal cells). The preview draws the new cell dashed.
- **Slab from Bulk…** (Ctrl+Alt+H): (hkl), the number of **atomic layers**
  (planes of atoms within 0.15 Å along the surface normal; ase's own `layers`
  counts surface cells), the termination (told apart by the composition of the
  top plane), and the vacuum, which is the total gap between periodic images.
  Terminations are named by the top plane's reduced formula ("O-terminated",
  "MgO-terminated"). While the dialog is open the slab preview covers the
  viewport, at its standard view with the vacuum running up, framed in the part
  the dialog leaves free; the bulk's own view is unchanged. The vacuum only matters
  for a 3D periodic cell: CRYSTAL's SLAB keyword (2D) ignores it, and the
  dialog says so.
- **Point Defect…**: vacancy, substitution, interstitial and antisite, on a
  site picked in the list (or from an atom's context menu), in a supercell
  searched so the defect's periodic images are at least Settings › Defect
  image distance apart (10 Å by default). Interstitials are found on a grid
  and snapped to their site symmetry, with Wyckoff letters. "One structure per
  site" makes one document per inequivalent site. A charge is kept with the
  document as a hint for Setup.
- **Transform Cell…**: primitive, conventional, or Niggli-reduced.
- **Symmetrize…**: the space group found at each of 13 tolerances from 10⁻⁵
  to 0.1 Å, and spglib's refined structure at the one picked, with arrows
  showing how far each atom moves. The result is the group's conventional
  cell: on a supercell that is smaller (MgO 2×2×2 becomes the 8-atom cell),
  and the dialog warns before you accept.

A preview that reaches past the view (a supercell's ghost atoms, a defect's
supercell outline) zooms the view out until it fits beside the dialog; Cancel
puts the view back, unless you moved the camera meanwhile.

**Limits.** In place stops at 5000 atoms (the editing limit). A new tab over
5000 atoms opens view-only, with the reason; over 20,000 atoms its space
group is computed on request (**Compute now**, a task). Builds stop at
250,000 atoms and refuse before building. Saving a supercell as a
**symmetrized** CIF is refused: spglib's refined cell would be the space
group's smaller cell, so Save As says to save P1, which keeps every atom.

## Files

- **Drag and drop.** Drop CIF, POSCAR/CONTCAR, XYZ, extXYZ, `.d12` or `.out`
  files anywhere on the window (up to 8 at a time, 50 MB each). Each opens as
  an unsaved document with its own name ("MgO.cif (dropped)" when the folder
  already has that name), marked "Not saved in the folder yet"; the server
  reads it from a temporary file it deletes at once, and nothing is written
  into the folder until you Save As. Folders, other extensions and bigger
  files are refused with a message.
- **Import from Clipboard** (Ctrl+Shift+V, or File menu): the text of a
  POSCAR, CIF, XYZ or extXYZ, with its format detected as you type ("Looks
  like POSCAR, 8 atoms") and, for a plain XYZ, a box with 10 Å of vacuum or a
  view-only molecule.
- **Copy, Paste, Cut and Duplicate.** Ctrl+C copies the selected atoms;
  Ctrl+V pastes them at the same fractional coordinates as one History entry
  ("Paste 3 Atoms") and selects them. In Sym mode each pasted atom brings its
  orbit, and more than three times the copied count asks first. Ctrl+X is
  Copy plus one Delete; Ctrl+D duplicates the selection one step to the right
  and grabs the copies (Esc leaves them there). Other text pasted into the
  window opens Import from Clipboard.
- **Export Structure** (File menu) downloads CIF (P1 or symmetrized), POSCAR,
  XYZ or extXYZ to this computer (over a tunnel: the laptop); **Save As**
  writes into the served folder. Export never writes into the folder.
- **Recent files** are kept per folder: File › Open Recent, the welcome
  screen and the explorer's collapsible RECENT section. Right-click one to
  open, pin (pinned files stay on top and don't count toward the limit),
  remove it, or copy its path. A file that is gone shows "Missing". Clear
  Recently Opened can be undone for 8 s. Built and dropped documents join the
  list when they are saved.
- **Changed on disk.** When the window regains focus and an open file changed,
  a banner offers Reload or Ignore.
- **Big outputs.** A CRYSTAL `.out` is indexed once (byte offsets of each
  geometry block, the energies and the OPT status, read by the same block
  parser); the Optimization view opens from a summary (energies, labels, the
  first and last steps) and loads other steps as you scrub. A 34 MB run opens
  in under 2.5 s and sends under 300 KB. The explorer shows each structure
  file's formula and atom count, read in the pool once the page is idle.

### Open Folder

**File › Open Folder…** (also in the command search, and the folder button in
the Explorer's header; no default key) serves another folder in this window
without restarting the server. Pick a root (Home, Scratch), then a folder, or
paste a path, and press Open; the window reloads into the new folder. Modified
documents ask first (Save All, Don't Save keeps them as drafts in this
browser, Cancel); drafts and recent files are kept per folder, so going back
offers them again. Other tabs of the same server follow with a notice. While
GUI tasks run, Open Folder waits ("Cancel or wait for the running tasks
first").

- **Which folders.** The roots are your home folder and your scratch folder:
  `$SCRATCH`, `$WORK` or `$PSCRATCH` when set, else the first of
  `/scratch/$USER`, `/mnt/scratch/$USER`, `/global/scratch/$USER` or
  `/work/$USER` that exists and is yours. `--root DIR` (repeatable) or
  `MACE_GUI_ROOTS` (a `:`-separated list) replace them; `--root none` (or
  `MACE_GUI_ROOTS=` set to nothing) turns Open Folder off. The folder you
  started with (`--dir`) is always allowed. A symbolic link that leads out of
  the roots is refused.
- **Never hidden or credential folders.** A folder with a hidden part below
  the root (`~/.config`, `~/proj/.git`) is refused: "Hidden folders cannot be
  opened here (they hold settings and credentials). Start mace gui with --dir
  to serve one on purpose." So is a folder holding `.ssh`, `.gnupg`, `.aws`,
  `.kube`, `.docker`, `.config/gh`, `.netrc`, `.git-credentials` or `.pgpass`
  (your home folder itself): "Open a project folder inside your home, not your
  home folder itself." Even when you serve your home folder with `--dir`, MACE
  GUI never reads those credential files or the other secret files listed in
  [Security](#security), and it never writes into a hidden folder.

### What each file opens as

The explorer lists every kind of file the CLI reads or writes, and decides
what an output or a log is from its content, not its name: the first 256 KB
and the last 64 KB are read in the pool (once per file version), and the
calculation type comes from the CLI's own `determine_completed_subtype`.
Each row's chip shows it (FREQ, OPT, SP, BAND, DOSS, CHG+POT, TRANSP, log, ...),
and the **Logs & data** filter gathers job logs, scripts, restart data and
wave functions.

| File | Opens as |
|---|---|
| CIF, `.d12` (3D) | a structure document |
| `.xyz`, `.extxyz`, POSCAR/CONTCAR/`.vasp`, `.mol`, `.sdf`, `.pdb`, `.gen`, `fort.34` | a structure (a file with no cell is a molecule, drawn with no cell, view only) |
| CRYSTAL `.out` (crystal, Pcrystal), also when named `.o` or `.log` | OPT: the optimization trajectory; SP: the structure; FREQ: **Normal modes** (Settings › Open FREQ outputs as: Structure brings back the structure); molecules too (their Angstrom geometry tables) |
| a CRYSTAL `.out` that stopped before printing a geometry | its text, with a Problems row saying so |
| properties `.out` (Pproperties: BAND, DOSS, CHARGE+POTENTIAL, TRANSPORT) | a properties document: the run's type and status, its data files (each opens its plot or volume), and the structure from the SCF output beside it |
| SLURM job logs (`<job>-<id>.o`, `<job>.o<id>`, array logs `<tag>-<A>_<T>.o`, `slurm-<id>.out`, `.e`, `.err`) | read-only text with a header naming the job and its calculation (**Open**); a `.e`/`.err` adds its first line to Problems. A log is never parsed for geometry |
| `.d3` | text |
| `*.BAND.DAT`, `*.DOSS.DAT`, `*_IRSPEC.DAT`, `*_RAMSPEC.DAT` | plots |
| cube files: `*_DENS.CUBE`, `*_POT.CUBE`, `*_SPIN.CUBE`, `.cube`, and CRYSTAL's own names `DENS_CUBE.DAT`, `POT_CUBE.DAT`, `SPIN_CUBE.DAT`, `<input>_WAN_CUBE.DAT` | the Volume document (below) |
| `fort.25` / `.f25` | a 2D map (ECHG charge density, POTM potential: the MAPN block) opens as a contour plot of the A-B-C plane with its corners labeled, a log scale for a positive map, and the spin density as a second map; each map is named by the properties output of the same name beside it (CRYSTAL's fort.25 does not say whether a block is a density or a potential), else "2D map"; a BAND or DOSS fort.25 says so and opens the `.DAT` beside it |
| BOLTZTRA `*.SEEBECK.dat`, `*.SIGMA.dat`, `*.SIGMAS.dat`, `*.KAPPA.dat`, `*.TDF.dat` | a transport plot |
| `FREQINFO.DAT`, `BORN.DAT`, `HESSOPT.DAT`, `OPTINFO.DAT`, `SCFOUT.LOG`, ...; `.f9`, `.f98` | a note on what the file is (MACE uses wave functions for GUESSP restarts and properties runs), with **Reveal the run's output**; never read past 4 KB |
| `.sh` job scripts | text |

Ctrl-click (or Shift-click a range of) outputs to pick several without
opening them; right-click › **Compare Convergence** plots each run's energy
per atom at every printed geometry, above the lowest final energy of the
set, with each run's OPT END line in the legend.

### Notifications

Information and success notices close after Settings › Info notices stay (6 s
by default; a notice under the pointer or the keyboard stays). Warnings and
errors stay until you close them (the ×, Esc on the notice, or Dismiss All).
The same notice shown again counts up ("×3") instead of stacking, and more
than three fold into "+N more". Nothing is lost: every notice of the session
is in **Problems › All Notifications** (the bell in the status bar shows how
many are new), and warnings and errors also appear under Problems.

## Calculation Setup

**Calculation › Calculation Setup…** (Ctrl+Shift+R) opens the settings of the
structure in front: `mace convert` (cif2d12) for a CIF or a built structure,
`mace opt2d12` for a follow-up from a CRYSTAL `.out`, and `mace opt2d3` for
BAND, DOSS and the other properties runs. For an `.out` the calculation types
come in rows of three: SP, OPT and FREQ (opt2d12) above the properties of its
wavefunction (opt2d3); the arrow keys move through them.

- **The CLI is the source of the defaults.** The form lists the CLI's own
  questions, its menus and its defaults, read from the CLI at the moment you
  open Setup. A follow-up starts from the parent's settings (functional, basis,
  SPIN, tolerances), as `opt2d12` does on the command line, and its fields say
  "parent"; opt2d3's fields start from opt2d3's own defaults and say
  "default". Each changed field says what it was and has a Reset. Fields the CLI
  ignores for a choice are read-only and say why ("auto from the cell: SHRINK
  12 24" for cif2d12's mesh).
- **Preview While Editing** (`setupAutoPreview`, on by default) previews after
  each change; Ctrl+Enter previews by hand. A newer edit cancels the older
  preview. **Run Preflight on Preview** (`runPreflightOnPreview`, off by
  default) adds the TESTPDIM check to each preview.
- **opt2d3 and the wavefunction.** A properties deck needs the parent's `.f9`.
  The preview stages an empty stand-in with the CLI's chosen name (it never
  copies a big `.f9`) and Problems says so; the printed command copies the
  real one when you run it. Without an `.f9`, Problems shows the CLI's
  refusal.
- **seekpath.** Band paths come from SeeK-path, as on the command line; there is
  no static path unless you pick the CLI's own static option.
- **A dirty `.out`.** opt2d12 reads the geometry from the output file, not your
  edits: Setup asks whether to use the unedited parent or to Save As CIF and
  set up a new calculation. The same applies to a trajectory frame that is not
  the last.
- **P1 CIFs.** When a CIF is P1 but spglib finds more symmetry at 1e-3 Å (a
  structure built in the GUI and saved with every atom), Problems says so and
  offers **Save Symmetrized Copy…**. cif2d12 keeps P1 otherwise; nothing changes
  silently.
- **Saving.** *Own folder* writes the JSON and the command in a new folder,
  *In place* writes them beside the input, and *Write decks now* also runs the
  printed command (after checking the disk space; on a tmpfs it warns about
  RAM).
- **Undo.** Setup has its own history (the History panel's **Settings** chip);
  Reset All Settings goes back to the CLI's defaults in one undo step.
- **Copy CLI Command** (Ctrl+Alt+C) copies the printed command.
- **Run Preflight** without a serial `crystal` on the machine runs MACE's
  structure check alone and says so; a failure there reads "Structure check
  failed (CRYSTAL not run)", never "REJECTED", which is kept for CRYSTAL's own
  TESTPDIM rejection.

## Submit and Jobs

**Calculation › Submit…** (or right-click a `.d12`/`.d3`, or a folder header
in the Explorer: **Submit Folder…**) submits decks with `mace submit`, and
**Run on This Node…** runs them here with `mace run` (a dev node, no SLURM).
Neither has a default shortcut, and nothing is submitted until you click the
dialog's own **Submit** button: the dialog opens with **Cancel** focused, and
Enter never submits.

- **The dialog runs the real command first, in a copy of the folder.** It
  shows the exact command a shell user would type (`cd <folder> && mace
  submit ...`, with `echo N |` for a folder of D12 and D3 decks), the
  resources read from the job scripts (cores, memory, walltime, account,
  partition, `-N`, exclusions; a reused `<deck>.sh` says so), every file it
  will write, and what the jobs then write (`<deck>.out`, `<deck>.f9`). In
  the preview, read-only SLURM queries (node types for `--exclude-type`, the
  job-array limit, MaxSubmitJobs) reach your cluster; every state change
  (`sbatch`, `scancel`, `scontrol release`) is only recorded. Without SLURM
  the preview says it used default cluster limits.
- **Options** map to `mace submit` flags: *Submit as job arrays* and *Max
  running at once* (`--array --max-running N`, a folder only: Batch Setup
  collects decks into one), *Track in materials.db* (`--track`, enables
  `mace status`), *Then run the rest of a workflow* (`--progress TEMPLATE`;
  only templates that contain your decks' calculation type, never
  `interactive` or `custom`), *Walltime* (`--walltime`, checked by the CLI),
  *Exclude* none, these nodes or a node type (`--exclude-nodes` /
  `--exclude-type`), and *Regenerate job scripts* (`--overwrite-sh`). Up to
  20 decks the preview updates as you change them; above that, **Update
  preview**.
- **Conflicts need their own tick.** A finished `<deck>.out` or a `<deck>.f9`
  the job would replace: *Overwrite finished results*. A deck already queued
  (your squeue, or a submitted/pending/running row of the folder's
  databases): *Submit again although ... already queued*. An account that is
  not one of your SLURM accounts: *Submit with account ... anyway*. The
  server checks the ticks again, and refuses a submit whose preview is used,
  older than 15 minutes, or whose decks, job scripts, outputs, queue, site
  settings or CRYSTAL module changed since (409, "preview again"; the dialog
  then offers **Update preview**). While a submit runs, the dialog cannot be
  closed.
- **Site** shows the SLURM account and the CRYSTAL module the job scripts
  load, and where each comes from (see [Site settings](cli.md#site-settings-mace-site-sitetoml)),
  then one line each for the job size (Resources), the MPI launch, the
  scratch order, QOS, constraint and the CRYSTAL executables, with the
  source of each value (the values `mace site --json` prints). **Check** runs
  `mace site check` on the machine the server runs on, with the environment
  a submit gets, and lists each item with its fix; it asks `sbatch
  --test-only` and submits nothing. The GUI never edits `site.toml`.
- **The real run** is the same argv, in the same folder, with the environment
  `mace gui` was started with: no `nice`, no thread limits, none of the GUI's
  own variables, so `sbatch` gives the job what a shell gives it. Inside an
  allocation (`salloc`), the allocation's `SLURM_*` variables are left out.
  A submit cannot be stopped halfway ("Submitting... N s"); stopping the
  server waits for it (a second Ctrl+C stops waiting; the submit finishes on
  its own). Afterwards the job scripts are compared with the preview's; a
  difference (the cluster changed in between) is reported.
- **Run on This Node** previews with `mace run --dry-run` on the selected
  decks; where it cannot run (a login node, inside a SLURM job, no CRYSTAL)
  the reason is `mace run`'s own. The run is a task in Tasks; Cancel sends
  one SIGTERM and waits for `mace run` to stop its job (up to 25 s) before
  anything is killed. The run stops when MACE GUI stops.
- **Submitting and canceling jobs are off while MACE GUI listens on the
  network** (`--insecure-network`).

**Jobs** (bottom panel, **View › Jobs**, or the status bar's "Jobs: 2 running
· 1 pending") lists your own SLURM jobs: *This folder* (their working folder
is inside the served folder) or *All my jobs*, job arrays grouped as "Array
A: 1 running, 2 pending". Each row has **Open Output** (the job's
`<deck>.out`), **Show Log** and **Cancel**; *Finished (last 2 days)* lists
ended jobs from `sacct`. It asks SLURM (`squeue`) every
`cluster.pollSeconds` (Settings › Cluster › Job poll interval, 60 s, 30 to
300) while it is shown or while a job this session submitted is pending or
running, through the same cached answer as the Cluster view and `mace
cluster` (Refresh asks again at most once per 15 s); a job that leaves the
queue gives one toast with its `sacct` state (and a Problems entry unless it
completed), once, however many views noticed it.
**Cancel** runs `scancel <id>` after a confirmation that names the command,
for your own single jobs and array tasks only: a whole array (whether MACE
submitted it or not), a task range, another user's job and the job MACE GUI
runs in are refused with the reason.

## Batch Setup

**Calculation › Batch Setup…** (or right-click a folder header: **Batch
Setup…**) opens a "Batch: <folder>" tab that sets up many decks at once, in
four steps:

1. **Pick files:** the folder's *CIF files*, *Completed runs needing a
   follow-up* (SP, OPT, FREQ, BAND, DOSS, TRANSPORT, CHARGE, POTENTIAL), or
   *Failed runs*, optionally with subfolders. A file that is not offered
   says why: the follow-up is already beside it, in an earlier
   `batch_*/batch.json`, in the folder's or a workflow's database, or next
   in a workflow plan; FREQ needs a converged OPT; properties need the
   wavefunction; a name with spaces or shell characters cannot be submitted.
   Failed runs show what can be done; a tracked one shows the `mace recover`
   command (it prepares fixed inputs and job scripts for every failed tracked
   calculation of that database and prints the command that submits each).
2. **Settings for all files:** the Setup form of one representative file.
   Only the fields you change are applied, to each file on its own through
   the CLI, so every follow-up keeps its own parent's functional, basis and
   settings otherwise.
3. **Preview:** each file's deck, how many lines differ from its default,
   **Diff**, and **Override...** for one file only. Files the CLI refuses are
   shown with its words and left out.
4. **Write and submit:** a folder `batch_<calc>_<date>/` (name it as you
   like) with `options/` (the CLI's settings files, one per CIF group or per
   parent, and `template.json`), `inputs/` (cif2d12's CIF groups), `decks/`,
   `commands.sh` and `batch.json`. `bash commands.sh` in that folder's parent
   re-creates every deck with the same commands (`mace convert --batch`,
   `mace opt2d12 ... --yes`, `mace opt2d3 ...`). **Submit...** opens the Submit
   dialog on `batch_*/decks`, with job arrays ticked for two or more decks; it
   is submitted from the folder that holds the batch, so tracked rows join
   the parents' `materials.db` there (`mace status` in that folder lists
   them; a `--progress` workflow keeps its records in
   `batch_*/decks/.mace_context_<id>/`, so run `mace status` there for it).

## The Database document

Opening a `materials.db` (or **Analysis › Open Materials Database**) shows a
table that the server sorts, filters and pages: the page holds only the rows
on screen, so 100,000 materials scroll at the display's frame rate (the bench
measures 13 ms to the first page and at most 45 row nodes). The filter row
searches material ids and formulas and narrows by dimensionality, calculation
type and status, a space-group range, and a property the material has. Selecting a row shows its calculations and properties in
the inspector, with links that open the input and output files; a property
that several of the material's calculations have names its calculation, as
in `total_cpu_time (OPT)`. **Export
Table** (CSV or JSON) writes the filtered, sorted rows with the CLI's own
columns. The table keyboard: arrows, Page Up/Down, Home/End over the whole
result, Enter to open.

The Database view reads the database; Create or Update Database from a
Folder writes to it, with a backup first. The view never waits behind a job:
while a completion callback records a result, the status line says "Database
busy (a job is recording its result) - retrying" and keeps the rows it shows;
a database left mid-write by a killed job says so until the next MACE command
that writes it repairs it. A database in a folder the server cannot write is
read as a snapshot, with a banner. While the view is visible it looks at the
file every 30 s (Settings › Files and data › Refresh the Database view: Off, 15, 30 or 60 s) and
reloads the page in place, keeping the sort, the filters and the selection
("Updated 4 s ago · 3 new calculations").

### Building a database from a folder

**Create or Update Database…** reads a folder's CRYSTAL outputs into a
`materials.db`. It is on the Database view's toolbar (**Update from
Folder…**), on the empty Database page (**Create Database from Folder…**), on
a folder's right-click menu in the Explorer (**Create or Update Database from
This Folder…**), in Analysis and in the command search. It runs the same
command a terminal would, shown at the bottom of the dialog:

```bash
mace analyze --extract-properties runs --db-path runs/materials.db --register --skip-unchanged --backup --keep-backups 3
```

- **Folder** and **Database**: the database is the open one, else the
  folder's `materials.db`, else the workflow's own database (one
  `.mace_context_*` folder), else a new `<folder>/materials.db`. The dialog
  says "Will update runs/materials.db (1,240 property rows)" or "Will create
  runs/materials.db". A database outside the served folder is not written
  from the GUI: the dialog shows the command to run in a terminal instead.
- **Register calculations (recommended)** (`--register`) records the
  folder's completed calculations first, so the Database view lists their
  materials. **Only new and changed files** (`--skip-unchanged`) extracts
  only what changed since the last build (size or content, never the file
  time alone). **Back up the database first** (`--backup`) is always on for an
  existing database: a consistent copy goes to `backups/` beside it, and the
  last 3 are kept (`--keep-backups`); the dialog lists them and says which one
  the update would prune.
- **Preview** is the command's `--dry-run --json`: every output is New,
  Changed, Unchanged, Not tracked (always re-extracted), Unreadable or
  Incomplete; click a chip to list those files.
- **Run**: a task with progress, the current file, the elapsed time and
  Cancel. Extraction commits file by file, so a canceled build keeps what it
  wrote and names the backup. When it ends: "Database updated: 12 new, 3
  changed, 0 failed · Open Database"; outputs that gave no properties are
  listed under Problems.
- While a build runs, an open Database view of that database shows
  "Updating… (build in progress)" and reloads when it ends. A second build or
  a tracked submit of the same database from the GUI waits for it.

Nothing is deleted from the database. Re-extracting a calculation replaces its own earlier property values; rows filed under an old material id for that calculation are moved, not removed. A backup is made first.

## Analyzing a run

Opening an optimization or single-point `.out` shows the run in the 3D view
with the Timeline (the energy card and the frame slider) in the bottom panel,
and a **Run** card first in the inspector:

- **How the run stopped**, in one line with an icon (green: converged or a
  normal end; amber: CRYSTAL stopped with a warning; red: failed). The words
  are those of `mace status`/`mace check`, refined by how the optimization
  ended: "Converged: all four criteria met at point 4 (CRYSTAL23 manual
  p.188)", "Stopped after the first point: the gradients were already below
  threshold", "Stopped with a warning: gradients converged and the energy
  change was below TOLDEE, but the displacements were not (manual p.208)",
  and so on. **Show in Output** opens the output at that line.
- The CRYSTAL version, calculation type, dimensionality, space group, atoms,
  functional, basis, SHRINK, TOLINTEG, spin, MPI processes, start and end,
  wall and CPU time, file size, SCF cycles per point and the final energy in
  eV and eV/atom: what the database would store for this output.
- **Copy CLI** copies the two terminal commands that print the same:

  ```bash
  mace analyze --convergence runs/mat1_opt.out      # add --json for the numbers
  mace check runs/mat1_opt.out
  ```

**The energy card** counts CRYSTAL's optimization points (the x axis is the
point number as the `.out` prints it) and draws point 1 on the same energy
surface as the others: CRYSTAL prints point 1's `TOTAL ENERGY(DFT)` without
its D3/gCP correction and compares later points with the corrected value, so
the card uses the corrected one (the energy table's tooltip shows both). Its
row has Log |ΔE|, Relative to (First or Final), Units, Copy TSV and **Open as
Plot**.

**Run Analysis** (Open as Plot, or Analysis › Run) is a plot document with
five tabs:

| Tab | Shows |
|---|---|
| Energy | E − E_ref per point (default: the final energy). A FINALRUN restart re-references the integrals, so each restart starts a segment with its own reference, marked by a dotted line. **Log Y** shows CRYSTAL's printed energy change per point (`DE`, red when it went up) and each point's distance to its segment's final energy, with the TOLDEE line. |
| Convergence | The maximum and RMS gradient and displacement per point on a log axis, each with its threshold (dashed) and where the threshold came from (the `.out`, the `.d12`, or the CRYSTAL23 default and its manual page). Filled markers met their criterion. Gradients and displacements are in CRYSTAL's internal coordinates for cell optimizations ("given without units", as CRYSTAL says). |
| SCF | SCF cycles per point, and |ΔE| per SCF cycle where the `.out` prints it: CRYSTAL prints the cycles of the first point only and moves the rest to `SCFOUT.LOG`, which stays in the job's scratch directory. |
| Cell | a, b, c, α, β, γ and the volume per point, absolute or in % of the first. |
| Tracked | Distances and angles followed through every frame (below). |

A click on a point selects that frame everywhere (the slider, the energy
card, the 3D view); the selected frame is a vertical line on every tab.
**Compare Runs…** overlays a second run; with different atom counts the
energies default to per atom, and absolute energies are offered only for the
same composition.

**The Atoms, Coordination and Measurements panels** of a run show the frame
on screen, read-only, under "Frame 3 of 4 · POINT 4 (read-only). Use Edit This
Step to change it." Moving the slider updates them. In Measurements, pick 2
to 4 atoms in the Atoms table (Ctrl+click) and **Measure Selected**, or type
the atom numbers; **Track Along the Run** adds the distance or angle to Run
Analysis › Tracked, followed across cell faces (an atom that leaves through one
face and comes back through the other does not jump).

**Problems** lists what the output says, with links to the line: how the run
stopped, SCF that did not converge, FINALRUN restarts (information when the
run converged), CRYSTAL's own WARNING lines (counted once each), tolerances
that differ from CRYSTAL's defaults, and imaginary frequencies of a FREQ run
beyond numerical noise (Settings › Plots › Imaginary-mode noise level, 20 cm⁻¹ by
default; smaller ones are listed as information), with **Show Modes**.

### Runs and files that are still being written

An output a job is still writing is shown as **live**, never as broken: MACE
reads it up to a snapshot and ignores a half-written last line, so a number
is never read half-printed. The run card says "Live · updated 40 s ago" and,
with **Follow the run** on, new points appear every 15 s without resetting
the zoom or the selected frame. A run counts as live while it has no end
marker and SLURM says the job is running, or (when MACE cannot ask SLURM) it
changed in the last 10 minutes (Settings › Files and data › Live window). A
run that looks stalled is still checked every 60 s until CRYSTAL's end marker
appears ("Finished"). The text view of a live output has **Follow (tail)**;
an output that a resubmitted job started over says "This output was replaced
(a new run started) · Reload". A BAND.DAT, DOSS.DAT or cube a job is still
copying opens with **Retry** instead of an error, and Calculation Setup waits
for a running job's wavefunction (`.f9`) before it writes decks that need it.

## The Cluster view

On a cluster (where `sinfo` or a quota tool exists) the activity bar has a
**Cluster** button (also **View › Cluster** and the command search; Settings ›
Cluster › Cluster view turns it on or off). It opens one document with five
tabs (six where a module system exists), all read-only and served from the
same rate-limited cache as [`mace cluster`](cli.md#mace-cluster):

- **Overview:** your running and pending jobs against your limits, cores in
  use against the QOS cap, fairshare, the fullest storage space, upcoming
  maintenance, and your jobs with a **Details** button each.
- **Nodes:** nodes by CPU type or partition (idle, mixed, allocated, down or
  drained, free cores, memory and GPUs) with a bar per row; **Show all
  partitions** adds the ones your accounts cannot use.
- **Will It Start?:** cores, nodes, memory, GPUs, constraint, partition and
  time; **Find Free Nodes** shows where it fits now, **Predict Start** asks
  SLURM (`sbatch --test-only`, nothing is submitted) and shows the predicted
  start in your browser's time zone. A job script of the folder can be
  predicted instead of a request.
- **Storage:** a Space and a Files bar per quota, amber from 80 % and red
  from 90 % (the `[quota]` levels of `~/.config/mace/site.toml`). **Find What
  Uses My Files…** runs the throttled scan of `mace cluster files` as a
  background task (with Cancel) and lists the busiest folders and cleanup
  commands to copy; nothing is deleted.
- **Limits:** your associations, current use and fairshare, and what the
  next submission of N jobs x C cores would run into.
- **Software** (only where `module` exists): the CRYSTAL module the job
  scripts MACE writes would load (the same answer as `mace site`, with the
  modules it passed over and why) and the xtb module Clean would load. The
  first look runs the module command once per code, which can take up to a
  minute on a busy login node; **Refresh** looks again.

The strip at the top says how old the data is; **Refresh** asks again, at
most every 15 s (the button counts down). When storage reaches the warning
level, a chip in the status bar ("scratch files 91 %") opens the Storage tab.

### Job details

**Details** on a job opens a Job document: state and reason, node, elapsed
time against the limit, a walltime warning with what MACE will do if the job
times out, progress from the live output (an optimization's four criteria
against their thresholds and "Point 14 of about 20-24"; a frequency run's
displacements), the time left as nested levels (the total, the stages of the
run and the loop running inside the current one, as in
[`mace cluster job`](cli.md#one-job-progress-eta-and-walltime)) with a "Why
this estimate" disclosure (what each level is based on and any threshold that
differs from CRYSTAL23's default; it stays open while the document
refreshes), cores, memory and CPU efficiency (or "Not measured by SLURM
for this job"), **Show Live Use** (one `sstat`), for an array task the
array's tasks in the queue (each with **Details**), and the job's files. A file
inside the served folder opens here; one in another folder of yours (your
home, your scratch) gets "Open in a New Window Rooted at …" with the command
to copy; anything else is only named ("Outside the folders this GUI may
open"). While the document is visible it refreshes with the job poll
(Settings › Cluster › Job poll interval, 30 to 300 s, default 60).

**Notify me when my jobs finish** (Settings › Cluster, off by default) shows
a browser notification when one of your jobs leaves the queue; clicking it
opens the job. The browser asks for permission first; it works on
`localhost` (the tunnel), not on a plain `http` address of another host.

## Plots: band, DOS, IR, Raman and FREQ modes

Plots are documents with their own options (with their own undo) in the
inspector. plotly loads from the installed package the first time a plot
opens, once; nothing loads from another site.

| Document | Opens from | Options |
|---|---|---|
| Band structure | `*.BAND.DAT`, Analysis › Plot Band Structure | energy window, Fermi shift (or drag the Fermi line), line width, spin (both, α, β) |
| DOS | `*.DOSS.DAT`, Analysis › Plot Density of States | energy window, Fermi shift, line width, spin, which projections are shown |
| IR spectrum | `*_IRSPEC.DAT`, Analysis › IR Spectrum | wavenumber window, normalize to the highest peak, **average** the conformers beside it (as `mace plotting --spectra --average`); the main peaks are labeled |
| Raman spectrum | `*_RAMSPEC.DAT`, Analysis › Raman Spectrum | as IR, plus the component: total, parallel, perpendicular or one of the six tensor components, or all |
| Vibrational modes | a FREQ `.out`, Analysis › Vibrational Modes | the mode table (Modes panel), amplitude, speed, arrow scale, Play, Export Mode as XYZ |
| IR/Raman from the modes | Modes › IR or Raman (or Analysis › IR/Raman Spectrum) on a FREQ output with intensities but no SPEC.DAT | peak width (FWHM), line shape (IR Lorentzian or Gaussian; Raman Lorentzian share η) |
| Transport | `*.SEEBECK.dat` and its siblings | the quantity (tabs), temperature, tensor component (average of the diagonal, xx, yy, zz) |
| Convergence | Explorer › Compare Convergence | line width, logarithmic axis |

The FREQ document animates one normal mode at a time in the viewport, at most
30 frames a second and never while the tab is hidden. The arrows follow their
atoms while the mode plays (one set per animation frame), and the Arrows box
turns them on or off without stopping the animation. The header's
**Structure** button opens the output's structure, and **IR** / **Raman**
open its spectrum. With Settings › Reduce
motion on it shows a still view: the atoms at rest, each with an arrow from its
surface along the mode, and a Play button. Arrows are turned into the drawn
cell's frame (for a slab, by matching a, b and their normal); when CRYSTAL's
frame cannot be matched the GUI says so.
The modes come from the output's FREQ section only: the output index records
where it is, and a worker parses that range with `mace plotting`'s own parser,
so a large output is never read whole. The table lists frequency, irrep, IR
and Raman activity; imaginary modes are flagged.

The rest of the band and DOS notes:

- Energies are in eV relative to the Fermi level. When a POTC run of the same
  material sits beside the data (`<material>_POTC.POTC.DAT` and
  `<material>_POTC.out`), the plotters put everything on the vacuum scale. The
  axis then reads E − E<sub>vac</sub>, and the dashed Fermi line sits at that
  shift instead of 0.
- CRYSTAL writes two spin blocks even for a closed-shell run. When the beta
  block equals the alpha block (bands) or its negative (DOS), the run is
  closed-shell and one set is drawn, with no spin legend. A spin-polarized run
  draws α solid and β dotted.
- k-point labels come from the `.d3` path. At a jump in the path (`U|K`) the
  line breaks, so no band is drawn across it.
- Without a `.d3`, the DOS columns are counted from the DOSS.DAT header
  (`NPROJ`).
- Only bands within 25 eV of the Fermi level are sent, at most 400 per spin;
  the plot says when bands were left out.
- `api/plot` still makes `mace plotting`'s static PNG/SVG, for scripts.

The IR and Raman notes:

- Every IR- or Raman-active mode is a marker on the curve: hover names it
  ("Mode 37 · 1234.5 cm⁻¹ · B2u · IR 123.4 km/mol") and a click opens Normal
  modes at that mode (a click on the curve picks the nearest active mode within
  half the line width). The FREQ output beside the SPEC.DAT is read for its
  modes when it was not read yet. A click on the Modes panel's stick spectrum
  selects that mode too.
- A spectrum broadened from the modes uses CRYSTAL's own width: IRSPEC and
  RAMSPEC's `DAMPFAC` is a full width at half maximum, 8 cm⁻¹ unless the run's
  `.d12` sets it, so the curve has the width of a SPEC.DAT beside it. Each line
  has unit area times its intensity, as CRYSTAL's curves. The HTML viewer of
  `mace plotting --freq` uses γ = 8 cm⁻¹ as a *half* width (FWHM 16 cm⁻¹), so
  its peaks look twice as wide; the CLI is unchanged.
- A degenerate set (a row such as `4-   6` in CRYSTAL's mode table) is one
  peak: CRYSTAL's IR intensity already includes the degeneracy
  (INTENS_n = ... d_n (Z_x² + Z_y² + Z_z²), as the output prints), so the
  GUI does not add it once per mode. The HTML viewer of `mace plotting --freq`
  still sums every mode of the set (a triply degenerate peak three times too
  tall); the CLI is unchanged.
- Raman intensities are relative, normalized by CRYSTAL to a maximum of 1000;
  from the modes the powder total is drawn (the SPEC.DAT plot offers every
  column).
- A FREQ output with no IR or Raman intensities says "This output has no IR or
  Raman intensities (run FREQCALC with INTENS)".

### Plot options

Every plot (band, DOS, IR, Raman, FREQ modes, the run card and Run Analysis)
has the same toolbar: **Log Y**, **Units** (eV, meV, Ha, kJ/mol, kcal/mol for
energies; cm⁻¹, meV, THz for frequencies), **Export** (PNG, SVG, Image… at a size and
DPI, CSV, Copy as TSV, Publication Figure…) and **Plot Options**: Log X, Relative to, Per (cell,
atom, formula unit), the axis ranges, the line width and Reset Zoom. A log
axis of values that are zero or negative plots |value| and says so. Each plot
document remembers its own choices; new plots start from Settings › Units and
numbers › Energy unit and Settings › Plots. Copy as TSV copies exactly what is
plotted (units, normalization, reference and range), with a header row.


## Volumes (cube files)

A cube opens as a **Volume** document: isosurfaces drawn in the page by 3Dmol
from a grid the server cut to Settings › Volume quality (Low 0.25 M, Medium
1 M, High 2 M points; the chip says "100×100×792 → 50×50×396 shown"). The
server reads the cube with `mace plotting --cube`'s own reader (`CubeFile`),
crops the vacuum as it does, and sends the voxel vectors as they are, so a
skewed (hexagonal) cell's surface sits on its atoms.

- **Surface**: level (a log slider for a density; ± for a signed volume), the
  + and − lobes (spin up and down, an orbital's two phases), opacity,
  wireframe, atoms, cell. The first level is the middle of the CLI's own list of
  isovalues, so the first view matches `mace plotting --cube`.
- **Slice**: a lattice plane (a, b or c) at full resolution as a contour map
  under the 3D view, resampled on Cartesian in-plane axes for a skewed cell,
  drawn as a plane in 3D; the arrow keys step it.
- **Statistics** come from the full grid: min, max and, for a density, the
  *integrated density on this grid*. With effective core potentials the core
  electrons are not in the density, so that number is the valence count; it is
  left out when the grid does not cover the cell (the density is cut off at a
  face, or a periodic grid spans less than the cell), and a periodic grid that
  repeats its first plane as its last counts it once.
- **Color by Potential…** (the header, or Analysis › Color Density by
  Potential) colors a density's surface by a potential cube of the same grid
  (the same stem's `_POT.CUBE` is offered first; a cube of another cell is
  refused with the reason). The automatic range is the CLI's: the potential
  sampled on the isosurface as `plot_dual_cube_isosurface` builds it (where
  scikit-image is installed; otherwise at the density's grid points within 2 %
  of the level), then ±max(|5th percentile of the negative values|, 95th of the
  positive ones). The colors are the CLI's too: red = positive (electron-poor),
  blue = negative (electron-rich). **Reverse Colors** (or Settings › Potential
  colors) gives the chemistry convention, red = negative; a custom range and
  Reset Range are in the inspector.
- **Difference…** (Analysis › Volume Difference) makes A − B (− C ...) with any
  coefficients and opens the result as an untitled volume; **Save Result As…**
  writes it as a new `.CUBE` with the CLI's `write_cube_file` and never
  replaces a file. Grids of different sizes or origins on the same cell are
  aligned by the exact affine map (each grid point sampled linearly in the
  other grid's own voxel coordinates): `mace plotting`'s `align_cube_grids`
  interpolates one point at a time in Python (hours for a 100 × 100 × 792 slab)
  and finds a point's index by projecting onto each voxel direction, which
  shears a skewed cell. Grids of one shape go through `perform_cube_arithmetic`
  itself. A slice of the result is a plane of the difference (read again from
  its operands), and Save Result As writes a temporary file that is linked to
  the name only if no file of that name exists.
- **Which CRYSTAL keywords give which grids**: ECH3 → `_DENS.CUBE` (charge
  density; with spin, also `_SPIN.CUBE`), POT3 → `_POT.CUBE`, properties LOCALWF
  with PRINTPLO (IPLOT < 0 and ICUBE > 0) → `<input>_WAN_CUBE.DAT`, the Wannier
  functions on a 3D grid, shown as a signed volume at ±0.2 (the level the
  manual's own example uses); ECHG and POTM → 2D maps (fort.25 MAPN, opened as
  contour plots). A WAN cube is recognized and drawn, but no real one has been
  tested yet.
- **Bonded minus superposed atoms** (the deformation density): run ECH3 on the
  SCF wave function, then PATO + ECH3 with the same grid records (PATO builds
  the density from superposed atomic densities), and take SCF − PATO with
  Difference. opt2d3 does not write a PATO deck yet.
- **Render with mace plotting** runs `mace plotting --cube` on the file, for the
  CLI's own figure (a difference made in the GUI has no file, so it is not
  offered there).
- A density is drawn with one surface: round-off values such as −1e-15 on its
  grid do not make it signed. A lobe with no surface at the current level is
  not drawn.

## Molecules, slabs and polymers

- A file with no cell (`.xyz`, `.mol`, `.sdf`, `.pdb`) opens as a molecule:
  no cell is drawn and it is view-only (the editor works on 3D cells).
- A periodic cell that holds a molecule, a slab or a polymer in vacuum (at
  least Settings › Vacuum gap, 8 Å, along one, two or three of its vectors)
  says so under the structure's badges ("Looks like a slab (vacuum 19.6 Å along
  c)"); a molecular crystal with no vacuum does not.
- Calculation Setup asks the CLI's dimensionality question (CRYSTAL, SLAB,
  POLYMER, MOLECULE). A structure with no 3D cell is answered for (MOLECULE,
  SLAB or POLYMER from its periodic directions). The CLI's coordinate writer is
  right only for orthogonal, correctly oriented cells, so the GUI first moves
  the structure rigidly (every distance kept): a molecule into an orthogonal
  cube (its largest extent + 2 × the vacuum gap), a slab with its two periodic
  vectors as a and b and c perpendicular to them (vacuum along z), a polymer
  along x with b and c orthogonal. A molecule or chain stored wrapped across
  the faces of its box (as a CIF keeps it) is made whole first (lattice
  translations only). The deck is the CLI's (`mace convert` through
  the answer driver) on that cell, so the GUI deck is the CLI's for the same
  oriented CIF. A cell whose vacuum is not along its vectors is refused with the
  reason. Save for such a source asks for a CIF first, as for any unsaved
  structure.

## Compare

One document for every before and after:

- **Analysis › Optimization Trajectory**: frame 0 against the selected frame,
  with the frame slider.
- **Analysis › Compare with Input**: an output against the structure it
  started from: the `.d12` of the same name, else the parent CIF its
  provenance names, else its first frame. A centered `.d12` (R, F, I) that
  CRYSTAL ran in its primitive cell is compared through the output's first
  frame, which is that deck as CRYSTAL read it (checked: same formula, volume
  per atom and space group).
- **Tools › Compare Two Structures**, or **Compare With Active** on a file's
  context menu in the explorer.

The frames of one run (every OPT frame, and Clean) are paired by index. Two
files are paired by position: each atom with the nearest atom of its element
under the minimum image, measured in fractional coordinates so two files in
different Cartesian orientations still pair (the Hungarian assignment when
scipy is installed, else greedy). Two files can list the same species in the
same order and still order the atoms of one element differently, so the
files' own order is kept only when it pairs the atoms as well as any other;
otherwise a banner says "Atoms matched by position (max mismatch 0.012 Å)".
Above 3000 atoms of one element the files' order is used as it is (the
distance matrix would take hundreds of MB). Different counts or species are
refused with the reason ("8 atoms vs 64: compare a supercell with Build ›
Supercell first").

The looks (Ctrl+Alt+O cycles them): **Overlay** (the initial structure as a
ghost; 35% opacity in light mode, 60% and lightened in dark mode so it stays
visible on the dark viewport), **Two-tone** (initial and final each in one
tone, element shading kept, with a legend), **Side by side** (two linked
views), and **Displacement map** (the final structure colored by \|Δr\| on a
viridis-like scale, with a legend in Å and scaled arrows).

The Timeline panel's sub-tabs hold the energy chart, Δcell, Δbond,
displacements and Δsymmetry ("Fm-3m (225) → Fm-3m (225)"). Click a header to
sort; **Copy TSV** copies the shown table; **Analysis › Export Table** writes it
as CSV or JSON. Analysis › Energy Along the Run, Bond Length Table and
Displacement Table open their sub-tab.

## Two editor groups side by side

Any two documents can sit side by side: an IR spectrum beside its structure,
two structures, two outputs, two volumes, two runs' plots, a database beside
a plot. Every document kind works in either group, with all its controls, and
either group can edit.

- **View › Editor Groups › Split Editor Right** (**Ctrl+\\**): a second group
  to the right with the active document (moved, so a file is edited in one
  place); with only one document open, the right group starts empty.
- **Open to the Side** on a file's context menu in the Explorer, **Move to
  Other Group** on a tab's context menu, **View › Editor Groups › Move
  Document to Other Group** (**Ctrl+Shift+\\**), or drag a tab onto the right
  group (with one group, onto the right half of the editor) to split.
  A document with unsaved changes moves after Save; one without a file (a
  comparison, a new structure) stays where it is.
- The focused group has a blue line over its tabs; click into a group, or
  **Focus Other Group** (**Ctrl+Alt+\\**), to focus it. Menu and keyboard
  commands that act on a document (Edit, Select, Build, Analysis, View Along,
  Save, Export Image, ...) go to the focused group, and the Explorer opens files
  into it. The status bar says "Right group" while the right group has the
  focus. The Inspector follows the focused group: it shows the right group's
  document (its plot options, volume surface, mode, cell and symmetry) while
  that group has the focus. Each group has its own bottom panel (atoms,
  timeline, modes, ...) under it. In a window narrower than 1100 px, where the
  Inspector is a drawer, the drawer shows the left group's document.
- **Link Cameras** (View › Editor Groups): two 3D views (structures, runs,
  volumes, modes) turn together; each keeps its own zoom and center.
- **Combined analyses** read the other group: a peak clicked in an IR or Raman
  spectrum opens its mode in the other group; **Color Density by Potential**
  takes the other group's volume as the potential without asking, and
  **Volume Difference** starts with the two groups' volumes; **Analysis ›
  Compare the Two Groups' Structures** compares them (as Compare With Active).
- Drag the line between the groups to resize them (or focus it and use the
  arrow keys). The split, its width, Link Cameras and the right group's
  documents are kept per folder in your profile and come back after a reload
  or on another port. **Close Group** closes the right group (it asks first
  when a document there has unsaved changes).

## Display range, boundary and rendering

Two Inspector groups, under **Display**, change the active structure document
only (they say "This document"); Settings › Viewport holds the defaults that
every document starts with, and a document follows a changed default for
every value it has not changed itself. **Display** (Cell, Bonds, Labels, the
style and colors) stays the same for all documents ("All documents").

**Range & boundary**:

- **Range**: from and to, in fractions of a, b and c (−5 to 10, the end at
  least 0.05 above the start). **Unit Cell** is 0 to 1, **Cell + 10%** is −0.1
  to 1.1 on each periodic axis, and **2×2×2** and **3×3×3** repeat the cell.
  An axis that is not periodic (a slab's c, a molecule) says so and ignores
  the range. The estimate below says how many atoms that draws; a range of
  more than 200 000 atoms is refused. View › Display Range › Custom… (and the
  toolbar's range menu) takes the same from and to values.
- **Boundary Atoms**: also draw the atoms on the faces, edges and corners of
  the range (Shift+A).
- **Bonded Atoms Outside**: **Bonded Neighbors** adds the atoms bonded to the
  drawn ones (one shell); **Whole Molecules** completes every discrete
  molecule that crosses the range. A chain, layer or 3D network is never
  grown: the group says the structure is an extended network (with its
  dimension, from the bonds MACE draws) and only completes discrete molecules.
  At most 50 000 atoms are drawn.
- **Polyhedra** (Shift+P), **Atom Size** and **Bond Radius**.

**Rendering**:

- **Preset**: Default (the look of earlier versions with subtle depth
  cueing), Flat, Publication (white background, outline, ambient light,
  ambient occlusion) and Presentation (black background). Changing a value
  afterwards shows Custom.
- **Depth Cueing**: how much the farthest atoms fade toward the background
  (15% by default; 0 is off).
- **Light Strength**, **Light Direction** (Front, Upper Left, Top; an
  off-axis light raises Ambient Light to 15% so the shaded side stays
  visible) and **Ambient Light** (light from all directions).
- **Ambient Occlusion** (darkens creases between atoms; off on very large
  structures), **Outline** with its width and color (Theme: the theme's
  text color, or black or white, whichever stands out on a white, black or
  custom background), and **Background**.

A dot after a label means the document has its own value; click it to use
the default again. **Reset** does that for the whole group, and **Save as
Default** makes the document's values the defaults. These options only
change what you see: decks, CIF files and the settings JSON never contain
them. Export Image (PNG) draws the document's look; Export SVG draws flat
colors (no lighting, depth cueing or outline).

In VESTA terms: VESTA's default boundary is 0 to 1 on each axis with the
atoms on the boundary included (here: Unit Cell with Boundary Atoms on, the
default); −0.1 to 1.1 is a common choice; "Search molecules" is Whole
Molecules.

The 3D views also keep every drawn atom inside the clipping planes after a
range change, a pan or a zoom.

## Coordination and polyhedra

**Tools › Coordination Analysis** opens the Coordination panel: for the
selected atoms, or for each symmetry-unique site (with its Wyckoff letter)
when nothing is selected, the coordination number, Hoppe's effective
coordination number (ECoN), the nearest and mean bond lengths, the neighbors
by element, and the neighbor shells. The inspector shows the same for one
selected atom ("CN 6 (O×6, 2.106 Å)").

Two atoms are bonded under Settings' bond tolerance. Metal-metal pairs count
only for an atom with no metal-nonmetal bond, so Mg in MgO has its six O and
copper its twelve Cu.

**View › Show Polyhedra** (Shift+P) draws the coordination polyhedra around
the cations (centers with 4 to 12 neighbors, all of another element): one
translucent mesh per element, with edges in the cell color. The geometry
worker builds them. Above 2000 drawn atoms they stay off and the viewport
banner says why.

## SVG export

**Export Image…** offers SVG beside PNG: MACE's own orthographic projector
draws one circle per drawn atom (boundary atoms and the display range
included), the bonds as two half lines in their atoms' colors, and the cell,
painted back to front. The drawing lines up with the screen (it uses the
viewer's calibrated projection) and has no external references. Perspective
is refused ("SVG export is orthographic; press 5"). On a Compare document,
Export Image writes the comparison as SVG, with one group per pane when side
by side. 8000 atoms take about 0.2 s, written in chunks so the page never
blocks.

## Run it locally

```
cd /path/to/your/calculations
mace gui                       # or: mace gui --dir examples/gui --open
```

It prints something like:

```
MACE GUI serving /home/you/calcs
  http://127.0.0.1:8765/s/Xq3…/?login=k8F…
  (the link signs in once; mace gui login-url prints another)
Token (for the sign-in form): 9vQ…
Press Ctrl+C to stop.
```

Open the link. It signs in **once**: the code in it becomes a session cookie
on the first load and the address bar drops it, so the same link from the
browser's history signs nobody in. For a new link run `mace gui login-url`
(`--port P` when several servers run; `--open` opens it), or paste the token
into the sign-in page. For a server started by `mace gui connect`, run `mace
gui login-url` on the computer where `mace gui connect` runs: the link uses
the forwarded port there (on the node it says so instead of printing a link). The `/s/…/` part is this session's own path; the
server answers nothing outside it.

- `--port P` (or `MACE_GUI_PORT=P`) picks the port; `--port 0` picks a free
  one for a one-off run. Browser drafts are kept per port, so if you rely on
  them use the same port every time (see [Drafts](#drafts)).
- `--open` opens a browser on this machine. On an ssh session with no display
  and no `$BROWSER` it opens nothing (a text browser would take over the
  terminal) and says so.
- `--verbose` (`-v`) logs each request on stderr (the token is never logged).
- `--root DIR` (repeatable, or `MACE_GUI_ROOTS`) sets the folders File ›
  Open Folder… may switch within (default: your home and scratch folders);
  `--root none` turns Open Folder off (see [Open Folder](#open-folder)).
- `--managed` is the mode `mace gui connect` uses over ssh: one
  machine-readable `MACE_GUI_READY` line on stdout, control lines on stdin,
  and the server stops when that connection closes and no task is running
  (see [First run from your computer](#first-run-from-your-computer)).
- When stdout is not a terminal (`nohup`, `sbatch`, `> gui.log`), neither the
  link nor the token is printed: the token goes to a state file only you can
  read, named in the output and removed when the server stops
  (`$XDG_RUNTIME_DIR/mace-gui-<port>.json`, or
  `~/.cache/mace/gui/url-<host>-<port>.json`, with the base URL, the token
  and the folder), and `mace gui login-url --port <port>` prints a sign-in
  link. `MACE_GUI_PRINT_TOKEN=1` prints the link and the token instead, for
  scripts. Scripts can also use the token in an `X-Mace-Token` header.
- `MACE_GUI_TOKEN` (or `--token`) sets the token instead of a random one per
  run. It must be at least 16 characters, each a letter, a digit or one of
  `. _ ~ -` (`python -c "import secrets; print(secrets.token_urlsafe(24))"`
  makes one); anything else is refused. `--token` on the command line is
  visible to other users in `ps`; `MACE_GUI_TOKEN` is not. A fixed token
  survives restarts (so does the session path derived from it).

To try it on the example folder shipped with MACE:

```
mace gui --dir examples/gui --open
```

## First run from your computer

The server runs where the data is (a cluster's dev node, a workstation); the
browser runs on your computer; an ssh tunnel joins them. Your computer needs a
current browser with WebGL2 (Firefox, Chrome, Edge or Safari) and an OpenSSH
client (it comes with macOS and Linux, and with Windows 10 and 11 as `ssh` in
PowerShell).

**Which method should I use?**

| You have | Use |
|---|---|
| MACE on your Linux or macOS computer (or WSL) | `mace gui connect` |
| Linux or macOS without MACE | `mace-gui-connect.sh` |
| Windows | `mace-gui-connect.ps1` (PowerShell). `py <MACE>\mace_cli gui connect` is expected to work too; checked by hand |
| MobaXterm | its local terminal + `mace-gui-connect.sh`, or MobaSSHTunnel |
| VS Code Remote-SSH | `mace gui` in VS Code's terminal, then Ctrl+click the URL |
| Nothing but ssh | the manual tunnel ([below](#run-it-on-a-cluster-and-view-it-on-your-computer)) |

### mace gui connect (Linux, macOS, WSL)

`mace gui connect` does everything in one ssh session: it logs in through
your cluster's login host (the *gateway*) to the node you name, starts `mace
gui` there in the folder you name, forwards a free local port to it, and opens
the browser with a sign-in link. It needs only Python 3 and ssh on your
computer (none of MACE's scientific packages).

```
mace gui connect --gateway me@hpcc.example.edu --node dev-node-01 --dir ~/runs
```

- **The first run** with no profile asks for the gateway, a jump host (blank
  for none), the node (blank runs on the gateway itself, which is shared: a
  dev node is better), the folder, setup lines (e.g. `module load Python`),
  the `mace` command on the node and a profile name, and saves them in
  `~/.config/mace/connect.toml` (Windows: `%APPDATA%\mace\connect.toml`, mode
  0600, no passwords). Later runs need only `mace gui connect` (the default
  profile) or `mace gui connect <name>`.
- **Flags override the profile** for one run: `--gateway`, `--jump`,
  `--node`, `--hop`, `--dir`, `--mace`, `--setup`, `--conda`,
  `--local-port`, `--ssh-option` (repeatable), `--root` (repeatable),
  `--no-open`. `--save [NAME]` writes them back; `--list` and
  `--forget NAME` manage the profiles; `--dry-run` prints the exact ssh
  command and exits.
- **The ~/.ssh/config way** works too: a `Host hpcc` block for the gateway
  (with `User`), and `--gateway hpcc`. `mace gui connect` asks `ssh -G` which
  user that alias logs in as, and uses the same user on the node.
- **Ctrl+C** stops the server on the node and its cache folder. When GUI
  tasks are still running (a local CRYSTAL run, a database build, a Clean),
  it asks first: "2 tasks are running on dev-node-01 (...). Stop them too?
  [y/N]". No (the default) leaves the server running only for them.
- **A dropped connection** (a closed lid, a network change) stops the server
  when no task runs. With tasks running it keeps them running and stops after
  the last one ends (plus 10 minutes with nobody reconnected; 24 hours at
  most). Run `mace gui connect` again with the same folder: it says
  "Reattached to the server still running on dev-node-01 (1 task running)."
- **Enter** in the terminal of `mace gui connect` opens a new sign-in link
  (each link works once).
- **Your local port** is `--local-port` (default 8765), so your browser's
  drafts come back on the next run; when it is busy, the next free one is used
  and it says so.

### Windows (PowerShell)

Windows 10 1809 and later ship `ssh`; `-J` (the jump through the gateway)
needs OpenSSH 8.1 or newer: check with `ssh -V` (the script warns when it is
older). Get the script on your computer in one of two ways: run
`mace gui connect --export-script powershell -o mace-gui-connect.ps1` on the
cluster and copy the file (your profile is filled in), or take
`scripts/mace-gui-connect.ps1` from MACE's repository. Downloaded files are
blocked by Windows: run `Unblock-File .\mace-gui-connect.ps1` once. Then:

```
powershell -ExecutionPolicy Bypass -File .\mace-gui-connect.ps1 -Gateway me@hpcc.example.edu -Node dev-node-01 -Dir ~/runs
```

The parameters match the flags of `mace gui connect` (`-Jump`, `-Hop`,
`-Mace`, `-Setup`, `-Conda`, `-LocalPort`, `-SshOption`, `-Root`, `-NoOpen`,
`-DryRun`), and it reads the same `connect.toml`. If no password or 2FA
prompt appears and the connection ends, set up an ssh key or agent, or use
`-Manual`, which prints the two-terminal commands of the manual tunnel. With
PuTTY instead, set the tunnel under Connection › SSH › Tunnels (source port
8765, destination `127.0.0.1:8765`) and connect to the node through the
gateway.

### MobaXterm

(a) Open MobaXterm's **local terminal** and run `mace-gui-connect.sh` (bash
comes with it): `bash mace-gui-connect.sh --gateway me@hpcc.example.edu --node
dev-node-01 --dir ~/runs`. Get the script with `mace gui connect
--export-script bash -o mace-gui-connect.sh` on the cluster, or from MACE's
`scripts/` folder.

(b) Or with **MobaSSHTunnel** (Tunneling): a new local port forwarding with
forwarded port 8765, SSH server the gateway (with your user), remote server
`dev-node-01` and remote port 8765. Start it, then in a normal MobaXterm
session on `dev-node-01` run `mace gui --dir ~/runs --port 8765` and open the
link it prints.

### VS Code Remote-SSH

Add the gateway and the node to your ssh config (Windows:
`C:\Users\<you>\.ssh\config`; Linux and macOS: `~/.ssh/config`):

```
Host hpcc
    HostName hpcc.example.edu
    User <user>
Host <dev-node>
    ProxyJump hpcc
    User <user>
```

where `<dev-node>` is the dev node you use (for example `dev-node-01`). Then
"Remote-SSH: Connect to Host..." `<dev-node>`, and in VS Code's
integrated terminal run `mace gui --dir ~/runs`. **Ctrl+click the sign-in link
it prints**: VS Code forwards the port and rewrites the link (a different
local port is fine). The Ports tab's "Open in Browser" opens the address
without the link, which shows the sign-in page; use the link from the
terminal (or `mace gui login-url`). `mace gui --open` uses VS Code's own
`$BROWSER`, so it opens on your computer.

### Jump hosts and nodes that accept logins only from the login node

- `--jump bastion.example.edu` (comma separated for several) reaches the
  gateway through a bastion: `ssh -J bastion,gateway node`.
- `--hop jump` (one ssh path from your computer to the node, through `-J`)
  needs the node to accept your own login (key, agent, password). If the
  node asks for 2FA too, you see two prompts.
- `--hop nested` makes the gateway log in to the node itself
  (`ssh gateway "ssh node ..."`), for clusters whose nodes accept logins only
  from the login node. The inner login cannot ask anything (it runs with
  BatchMode): it works with host-based logins or keys on the gateway. The
  forwarded port then also sits on the gateway's own `127.0.0.1`, where other
  users of the login node could reach it; the session token still guards
  every request there, as on the node.
- `--hop auto` (the default) tries `jump`; when the node refuses the login
  through the gateway, it tries `nested` once and says "connecting through the
  gateway instead (hop = nested). Save it with --save so this is not tried
  twice." (A 2FA gateway then asks twice on that first run.)

## Run it on a cluster and view it on your computer

The manual way, with two terminals: the server on the node, the tunnel on
your computer.

1. Log in to a **dev node** (or to a compute node in an `salloc`), load your
   cluster's Python 3.11 or newer (e.g. `module load Python`) and start the
   GUI:

   ```
   ssh hpcc
   ssh dev-node-01
   module load Python
   cd /mnt/scratch/$USER/my_calcs
   mace gui
   ```

   In an ssh session it names the exact node and prints the tunnel lines:

   ```
   This server runs on dev-node-01. The tunnel must go to this exact node.
   From your computer, with MACE installed there:
     mace gui connect --gateway you@<gateway> --node dev-node-01 --dir /mnt/scratch/you/my_calcs
   Or with ssh alone (leave it running), then open the URL above:
     ssh -N -J you@<gateway> -L 8765:127.0.0.1:8765 you@dev-node-01
   ```

2. On **your computer**, in a second terminal, run that `ssh -N -J ... -L ...`
   line (it prints nothing and keeps running; that is expected), then open the
   `http://127.0.0.1:8765/s/.../?login=...` link in your browser. The tunnel
   must go to the node `mace gui` names: another dev node behind the same
   gateway does not see the server.

If you have an `~/.ssh/config` entry for the gateway (for example `Host hpcc`),
`ssh -N -J hpcc -L 8765:127.0.0.1:8765 dev-node-01` does the same thing. If port
8765 is already in use on your computer, change the first number and use that
port in the link. The hint prints `<gateway>` unless you name your cluster's
login host with `export MACE_GUI_GATEWAY=<host>` (in `~/.bashrc`) or
`--gateway <host>`. On a shared node pick a personal port so you do not collide
with other users and your drafts stay on one port: `export MACE_GUI_PORT=8xxx`
in `~/.bashrc`.

### Why port forwarding and not X11

`ssh -X` with a GUI program draws every frame on the cluster and sends the
pixels to you. The HPCC dev nodes only have software OpenGL (llvmpipe), so a
3D viewer gets a few frames per second at best. With port forwarding, only a
few KB of JSON per structure cross the network, and your browser draws the 3D
view on your laptop's GPU. X11 still works as a fallback if you cannot open a
tunnel: run `firefox` on the dev node (if it is installed there) over
`ssh -X` and open the URL in it. It is slow, but it works.

### Sign out and stop the server

**File › Sign Out and Stop Server…** (also in the command search, "quit" or
"logout", and as Stop Server… in the connection dialog of the status bar)
ends your session and stops MACE GUI on the server, cleanly: its temporary
folders and state files go, the port is free again, and every open tab shows
"MACE GUI has stopped" with the command that starts it again. Modified
documents ask first (Save All, or Don't Save, which keeps them as drafts in
this browser). The dialog lists the GUI tasks still running; stopping cancels
them, while **Stop When Tasks Finish** keeps the server (and your session)
until they end, with a banner in every tab and **Cancel Stop** to undo it.
Your SLURM jobs keep running; only MACE GUI stops. "Also remove the drafts and
recent files for this folder" is for a shared computer. Through `mace gui
connect`, its terminal then says "The server on dev-node-01 was stopped from
the browser." and exits.

### Long sessions on shared nodes

Some clusters limit every process on a dev node to a fixed amount of CPU time
(check with `ulimit -t`; many allow 120 minutes). The kernel kills a process
that reaches it. MACE GUI's server is mostly idle, but its helpers can get
there, so when such a limit is set:

- it says so when it starts ("This node limits each process to 120 CPU
  minutes. ...") and shows a "CPU 62 of 120 min" chip in the status bar from
  half of the limit;
- it restarts the ML worker (the model reloads, a few seconds) and starts
  fresh pool workers before they reach it; a pool worker the kernel kills
  anyway fails only its own task ("This node's CPU-time limit (120 min per
  process) stopped the task."), and the other tasks run again;
- at 80 % it warns, and at 97 % the server stops itself cleanly ("stopped
  before this node's CPU-time limit"); start it again for a fresh limit;
- a database update the limit stops keeps every file it already wrote; start
  it again with "Only new and changed files" on (from a terminal, the same
  `mace analyze` command with `--skip-unchanged`) and it continues where it
  stopped. Continuing by itself is planned for MACE 1.1.6.

Only per-process limits (`ulimit -t`) are detected; limits enforced in other
ways (cgroups, a site's monitoring daemon) are not. For long, heavy work,
submit jobs instead of running them on the dev node, or run MACE GUI itself
as a job (next section).

### Run it as a SLURM job

For long or heavy sessions (ML Clean, large outputs, a database build), MACE
GUI can run as a small batch job on a compute node instead of on a shared dev
node:

```bash
mace gui connect --slurm --gateway me@hpcc.example.edu --node dev-node-01 --dir ~/runs
                                  # from your computer
mace gui --slurm --dir ~/runs     # on a dev or login node of the cluster
```

What happens:

1. `sbatch` gets one job named `mace-gui` (`--parsable`, one argument per
   option) that runs `mace gui --managed --http-control --ready-file ...` in
   the folder (the server writes its address and token to that 0600 file, not
   to the job's log), with
   the environment you started it from (sbatch exports it, so a module or
   conda Python keeps working) and `--propagate=NONE` (a dev node's
   per-process CPU-time limit is not carried into the job). The job script is
   written to `~/.cache/mace/gui/slurm/` (mode 0700) and removed once
   submitted; the session token is never in it, in the job's environment or
   in its log.
2. While the job waits: "Waiting for a compute node (job 123, pending:
   Priority, expected start ...). Press Ctrl+C to give up." MACE asks SLURM
   every 30 seconds (through the same rate-limited cache as the Jobs panel
   and `mace cluster`). Ctrl+C, or a closed connection, cancels that job and
   nothing else, also when it has just started and MACE GUI is not ready
   yet.
3. When it runs, MACE forwards a port of the node you are on to the server
   on the job's node (`ssh -N -L ... <job node>`). This needs ssh to the node
   of your own running job, which most clusters allow (pam_slurm_adopt). If
   yours does not, the job is canceled and MACE says so; run MACE GUI on a
   dev node instead. A port that is busy on the node you are on (a shared
   login node) is not that: MACE takes another one (`mace gui connect` does
   too) and keeps the job.
4. The status bar shows "Job 123 · until 14:00". At 15 and 5 minutes before
   the job's time limit a banner warns; one minute before, the server stops
   cleanly. With nobody using it, it stops after 1 hour (Settings > Server
   changes this; an idle allocation spends your fairshare).
5. Ctrl+C in `mace gui connect` or `mace gui --slurm`, or File > Sign Out and
   Stop Server ("This also ends the job 123 ..."), stops the server; the job
   then ends and its allocation is released. Once its server answers, a job
   is never canceled from your side: its server stops and the job with it. If tasks are still
   running when the connection drops, the server keeps them, and running the
   same command again reattaches to that job instead of submitting another.

The request (flags of both commands; `mace gui connect` also keeps them in
its profile as `slurm = true`, `slurm_time`, `slurm_cpus`, `slurm_mem`,
`slurm_partition`, `slurm_account`, `slurm_qos`, `slurm_constraint`,
`slurm_gpus` and `slurm_args`):

| Option | Default | Notes |
|---|---|---|
| `--time` | 4:00:00 | the job's time limit |
| `--cpus` | 2 | one or two pool workers; the server itself is mostly idle |
| `--mem` | 4G | the server and one ML model; raise it for ML Clean or large files |
| `--partition`, `--account`, `--qos`, `--constraint` | your site settings, else SLURM's defaults | the `[slurm]` values of `mace site` that you or your group set |
| `--gpus` | none | e.g. `1` for ML Clean on a GPU; without it the GUI's GPU setting defaults to None (CPU) in the job |
| `--sbatch-arg` | | one more sbatch option as `--name=value` (repeatable), e.g. `--sbatch-arg=--reservation=r1` |

Smaller requests start sooner and cost less fairshare. `mace gui --slurm
--dry-run` prints the sbatch command and the job script without submitting.
The standalone scripts in `scripts/` do not support `--slurm`; use `mace gui
connect --slurm`.

## Security

A dev node is shared: any user logged in there can connect to its
`127.0.0.1`, so only the session token protects your session. The server:

- binds `127.0.0.1` only. Another address needs `--host` together with
  `--insecure-network`, because the token and your data then cross the
  network **unencrypted** (plain HTTP); the ssh tunnel is the supported way.
  With `--host 0.0.0.0` it accepts the node's own names and addresses as the
  Host and prints them. The test hooks (below) are never enabled then;
- requires a signed-in session or the token on **every** request: the token
  is random (192 bits) unless you set your own, which must be at least 16
  URL-safe characters (letters, digits, `. _ ~ -`; see
  [Run it locally](#run-it-locally)). The session path is derived from the
  token, so a log that shows the path is a reason to choose a long one. The
  token is never accepted in a URL or a cookie (see
  [Sign-in links, sessions and expiry](#sign-in-links-sessions-and-expiry));
  the session cookie (`HttpOnly; SameSite=Strict`) holds a random session id
  and its path is the session's own `/s/<nonce>/`, so the browser never sends
  it to another server it reaches on `127.0.0.1` or `localhost` (a Jupyter or
  TensorBoard tunnel, another user's app on the same node). The server
  answers nothing outside that path;
- does not print the token when stdout is not a terminal (see
  [Run it locally](#run-it-locally)); `--token` is visible in `ps`,
  `MACE_GUI_TOKEN` is not;
- refuses requests whose `Host` header is not the loopback name. This guards
  against DNS rebinding;
- accepts a POST only with an `X-Requested-With: mace-gui` header, which a
  page on another site cannot send without a CORS preflight that the server
  never answers;
- only serves files inside `--dir`. Symbolic links that lead outside it are
  not followed, and no save writes through a symbolic link;
- sends a Content-Security-Policy that allows scripts from the page's own
  origin only (no `eval`: neither 3Dmol.js nor the SVG plots need it), and
  serves HTML and SVG plots with a sandbox policy, so a plot cannot read the
  API or the cookie;
- drops a connection that sends no request within 30 s and serves at most 64
  connections at once; a connection that never sent a valid token is the
  first to go, so idle sockets of another user cannot lock you out;
- never enables its test hooks (`MACE_GUI_TEST_HOOKS=1`, for MACE's own
  tests) by accident: it says so on stderr when they are on, refuses them on
  a non-loopback address, and they never return the environment.

### Sign-in links, sessions and expiry

- The link `mace gui` prints works **once**: reopening it from the browser's
  history (or a chat app's link preview that opened it first) shows "This
  sign-in link was already used or has expired." with a sign-in form (a
  browser that is still signed in just shows MACE GUI). New links: `mace gui
  login-url`, or Enter in the terminal of `mace gui connect`.
- The token is for the sign-in form and for scripts (`X-Mace-Token` header).
  Wrong tokens and codes are answered after one second each, one at a time
  for the whole server, so guessing is slow; the right token never waits.
- A session ends after 12 hours without use (Settings › Server › Sign out
  after: 1, 4, 12, 24, 72 hours or Never; `MACE_GUI_SESSION_IDLE_HOURS`) and
  at the latest 7 days after sign-in (`session.maxDays`, 1 to 30;
  `MACE_GUI_SESSION_MAX_DAYS`). Only
  keyboard and mouse input count as use: a tab left open, its polls and its
  event stream do not. The page then shows **Signed Out**: your edits are kept
  as drafts in this browser, tasks keep running on the server, and pasting the
  token signs you in again.
- **Stop the server after … without use** (Settings › Server,
  `MACE_GUI_IDLE_STOP_HOURS`; off by default) stops the server after that many
  hours with no input from any tab and no task running. Every tab shows a
  warning 5 minutes before; any click or key keeps it running.
- A fixed token (`--token`, `MACE_GUI_TOKEN`) survives restarts. The state
  file holds the token (mode 0600, removed at exit), as the URL file did.

### Open Folder, mace gui connect and VS Code

- Open Folder stays inside the allowed roots (your home and scratch folders,
  or `--root`), refuses hidden and credential folders, and `--root none`
  turns it off ([Open Folder](#open-folder)). Nothing is written into a
  hidden folder.
- Credential and secret files are hidden for safety, wherever they are below
  the served folder and even when you serve your home folder: they are not
  listed, and opening, previewing or downloading one (also through a
  symbolic link) answers "<name> is hidden for safety (it may hold
  credentials or other secrets); MACE GUI never opens it". They are:
  - folders `.ssh/`, `.gnupg/`, `.aws/`, `.azure/`, `.kube/` (with
    `.kube/config`), `.docker/` (with `.docker/config.json`), `.config/gh/`
    and `.config/gcloud/`;
  - files `.netrc`, `.git-credentials`, `.pgpass`, `.pypirc`, `.npmrc`,
    `.vault-token`, `.env` and `.env.*` (`.env.local`, ...);
  - shell and interpreter histories: any dot name ending in `history`
    (`.bash_history`, `.zsh_history`, `.python_history`, `.psql_history`,
    `.Rhistory`, ...), `.histfile` and `.lesshst`;
  - ssh private keys anywhere: `id_rsa`, `id_dsa`, `id_ecdsa`, `id_ed25519`,
    `id_xmss` and names that continue them (`id_ed25519_work`); `id_*.pub`
    is public and opens.

  The list lives in one place (`mace/gui/roots.py`). Only the first group's
  folders and `.netrc`, `.git-credentials` and `.pgpass` make a folder count
  as a home folder for Open Folder; a project's own `.env` or `.npmrc` does
  not.
- `mace gui connect` gets the token only through the encrypted ssh channel
  (`MACE_GUI_READY` on the server's stdout) and opens the browser with a
  one-time link. While a dropped connection leaves the server running for its
  tasks, a 0600 state file in `~/.cache/mace/gui/` holds the token so that
  `mace gui connect` can reattach; it is removed when the server stops. With
  `--hop nested` the forwarded port also listens on the gateway's own
  `127.0.0.1`; the token guards it as on the node.
- In VS Code, the terminal's Ctrl+click goes through VS Code's own encrypted
  forwarding; the Ports tab address needs the sign-in link.

**Drafts and the shared origin.** Browser drafts are kept per origin, that
is per `http://127.0.0.1:<port>` in your browser. Any other page served at
that same address in the same browser can read them, so do not reuse your
GUI port for other tunnels.

### What the GUI writes into your folder

| What | Where | Notes |
|---|---|---|
| Save, Save As, Save All, the classic Save CIF | inside `--dir` (subfolders too) | a temporary file, then a rename; `<file>.bak` the first time a file is overwritten; never through a symbolic link |
| Save JSON (deck settings) | `--dir` | the same rules |
| Calculation Setup's settings file, its CIF copy (`setup/<name>/`) and, when you ask, the decks | inside `--dir` | the same rules; the decks are written by the CLI command it shows, into a new private folder first, then moved in (a deck already there keeps a `.bak`) |
| Batch Setup › Write | a new `batch_*/` folder inside `--dir`, nothing outside it | a replaced deck keeps a `.bak` |
| Submit | exactly the files `mace submit` writes (`<deck>.sh`, `.mace_arrays/`, `workflow_configs/`, `.mace_context_*`, `materials.db` rows) | for a tracked submit, one backup of the folder's `materials.db` in `backups/` (the newest three are kept) and File History copies of the files the submit replaces; the jobs then write their `<deck>.out`, `<deck>.f9` and logs |
| Run on This Node | what `mace run` writes (`.mace_local/`, the outputs) | |
| Create or Update Database from a Folder | the database you choose inside `--dir` (its `-wal`/`-journal` while it is written, `structures.db` beside it) and `backups/` beside it | written by the `mace analyze --extract-properties` command the dialog shows; a consistent backup first (the last 3 kept); nothing is deleted from the database |
| File History: the previous version of every file a GUI write replaces | `~/.local/state/mace/gui/history/` (or `$XDG_STATE_HOME/mace/gui/history`, or `MACE_GUI_HISTORY_DIR`), never inside `--dir` | per folder and file; the last 20 versions of the past 30 days, 500 MB for inputs and 1000 MB for copies of outputs, never the newest copy of a file; files over 50 MB are not copied (their `.bak` stays). If the copy cannot be written, the GUI refuses the write |
| Settings | `~/.config/mace/gui.json` (or `$XDG_CONFIG_HOME/mace/gui.json`) | written atomically |
| The "built from" records of saved files (keyed by the saved bytes' hash) | `~/.cache/mace/gui/provenance/` (or under `$XDG_CACHE_HOME/mace/gui`, or `MACE_GUI_STATE_DIR`) | at most 2000 small files, oldest removed first; safe to delete |
| Clean's method check, GFN-FF and ML runs | `$XDG_CACHE_HOME/mace/preopt/` and `<tmp>/mace-preopt-<uid>/` | see [docs/preopt.md](preopt.md) |
| The Cluster view's cache (nothing in your folders) | `~/.cache/mace/cluster/<cluster>/` (or under `$XDG_CACHE_HOME`) | about 20 small files, mode 0600; shared with `mace cluster` |
| The state file (base URL, token, folder, for `mace gui login-url`; `mace gui connect` writes one for the forwarded port on your computer, and a server it starts writes one without the token on the node) | `$XDG_RUNTIME_DIR/mace-gui-<port>.json` or `~/.cache/mace/gui/url-<host>-<port>.json` | mode 0600, removed at exit |
| `mace gui connect` profiles | `~/.config/mace/connect.toml` (Windows: `%APPDATA%\mace\connect.toml`) | written only by the first run or `--save`; mode 0600, no passwords |
| A server kept for its tasks after the connection dropped | `~/.cache/mace/gui/managed-<host>-<port>.json` | mode 0600, removed when the server stops |
| The environment-module search (xtb) | `~/.cache/mace/modules.json` | 7 days; safe to delete |
| Plot renders, deck previews, Setup runs | `$TMPDIR/mace_gui_cache_*` and `$TMPDIR/mace_gui_deck_*` | removed when the server stops |

**Never written over:** calculation outputs (a CRYSTAL `.out` and SLURM job
logs), wavefunctions (`.f9`, `.f98`, `fort.9`, `fort.98`, `fort.20`), data
files (`*.DAT`, `fort.25`, cubes, `SCFOUT.LOG`), SQLite files, MACE's own job
scripts (recognized by content) and the deck of a job that is queued or
running. Save As to such a name is refused in the dialog with the reason.
Before any other replace, the current version goes to File History
(Explorer › Show File History…, or File: Show History in the command search:
Open, Compare with Current, Restore This Version); the save toast offers
**Undo** for 30 s. The `<file>.bak` beside a file is kept as before.

**materials.db and running jobs.** The Database view reads the database
through read-only connections and never waits behind a job. The GUI writes it
only through Create or Update Database (and the tracked submit), always after
a backup. On a local disk MACE keeps SQLite's WAL mode, as before; on a
network or cluster file system (NFS, GPFS, Lustre, BeeGFS, CIFS, …), where <!-- british-ok: Lustre is a file system -->
SQLite's WAL is not safe between nodes, every MACE command uses the rollback
journal instead, and on mounts without cluster-wide locks it also serializes
its writers with a lock file beside the database (it says so once). Set
`MACE_DB_JOURNAL=wal|delete` or `MACE_DB_LOCKING=trust|degraded` to override;
set them in `~/.bashrc` or not at all, so the jobs' completion callbacks use
the same mode as your shell.

After a crash or `kill -9`, the
`$TMPDIR/mace_gui_cache_*` folder and the state file can be left behind; both
are safe to delete. On HPCC `$TMPDIR` may be unset (then `/tmp`) or node-local.

## Dependencies

No new packages. The server uses only the Python standard library
(`http.server.ThreadingHTTPServer`); structures and the editor use ase and
spglib, and plots use plotly, all three from MACE's `requirements.txt`. If
one is missing (an install that never ran `pip install -r requirements.txt`
again), the feature that needs it answers with the `pip install` line
instead of failing, and the rest works. The process pool relies on two
internals of Python's `multiprocessing`; it is tested on Python 3.11 and
3.12, and on another version it falls back to the standard start method
with one warning.

The page uses:

- **3Dmol.js 2.5.5** (BSD-3-Clause), vendored with its license in
  `mace/gui/static/vendor/3dmol/`, beside `THIRD_PARTY_LICENSES.txt` for the
  MIT/zlib libraries its build bundles (iobuffer, netcdfjs, pako, upng-js);
- the VESTA and Jmol element colors in `mace/gui/static/element_colors.json`,
  taken from pymatgen's `ElementColorSchemes.yaml` (MIT License, The Materials
  Project / LBNL, commit 9100860d; license text in
  `element_colors.LICENSE.txt`; `mace/gui/tools/make_element_colors.py`
  regenerates the file from that YAML);
- the **Lucide 0.469** icons the shell uses, as one sprite
  `mace/gui/static/icons/lucide.svg` (ISC License, which covers the Feather
  portions under MIT; license text in `icons/LICENSE-lucide.txt`), next to
  MACE's own crystal glyphs in `icons/mace-icons.svg`;
- the MACE logo (`mace/gui/static/brand/`, copies of [docs/logo](logo/README.md); MIT, this repository);
- **plotly.js**, served from the `plotly` Python package that MACE already requires
  (`plotly/package_data/plotly.min.js`).

The page loads nothing from a CDN, so it works on nodes without internet access.

## Example folder (`examples/gui/`)

`examples/gui/build_examples.py` rebuilds this folder. Nothing in it is
research data, and no number is made up for the GUI:

- `structures/`: symmetrized CIFs for textbook crystals (MgO, LiF, NaCl, Si,
  diamond, hBN), built from experimental lattice constants;
- `outputs/`: CRYSTAL outputs copied from MACE's own test files:
  - Si SP: the phonopy CRYSTAL example, BSD-3; see
    `tests/fixtures/phonopy_crystal/README.md`;
  - an hBN slab SP: a short CRYSTAL-style excerpt of the textbook hBN
    monolayer written for MACE's tests, labeled as synthetic in its first
    lines, there to show a 2D slab;
  - an **unfinished** Ag2Br3 FULLOPTG run (the R-3c mp-862982 cell, which the tests call `tqc_agbr`) with 6 geometries (from
    `tests/data/opt_restart`). It is there to show the before/after view;
- `materials.db`: written with `MaterialDatabase` from what those files
  contain. It has 5 materials with OPT/SP calculations and their statuses. Every
  number in it comes from the files above.

No BAND/DOSS outputs ship in the folder, because the repository has no real
ones of a textbook system. To see the Results tab, point `--dir` at a directory with
`*.BAND.DAT` / `*.DOSS.DAT` files (and the `.d3`/`.out` of the same run, which
the DOS plotter reads).

## Limitations

- The editor works on 3D crystals. It cannot edit SLAB, POLYMER or MOLECULE
  structures, and it refuses ghost and dummy atoms, which a CIF cannot carry.
- The Volume document has no orbital viewer of its own (a WAN cube is a signed
  volume) and no GIF export (the cube engine makes one:
  `python mace/plotting/engines/crystal_cubeviz_plotly.py FILE.CUBE --save-gif out.gif`,
  with kaleido and Pillow; `mace plotting` does not pass `--save-gif` on).
- Two editor groups at most (Split Editor Right); a tab dragged from the right
  group goes back with Move to Other Group (its tab menu or Ctrl+Shift+\\).
  A gallery of many outputs in panes, cycled with a caption table, is planned
  for MACE 1.1.6; Compare Runs overlays convergence meanwhile.
- GUI preferences are per user; per-folder overrides (`.mace/gui.json` in a
  project) are planned for MACE 1.1.6.
- A database update stopped by a dev node's CPU-time limit does not continue
  by itself yet (planned for MACE 1.1.6); start it again with "Only new and
  changed files".
- Editing stops at 5000 atoms (viewing does not).
- The structure readers cover CIF, 3D `.d12` and CRYSTAL `.out`. SLAB, POLYMER
  and MOLECULE geometry is read from the `.out`. spglib symmetry is reported for
  3D cells only, because layer and rod groups come from CRYSTAL itself.
- The before/after arrows show the atomic relaxation, which is the change in
  fractional coordinates expressed in the final cell. The cell strain is listed
  separately in the cell table, because a Cartesian difference would mix the
  strain in, depending on where the cell origin is.
- The bond table and Clean count two atoms as bonded when they are closer
  than 1.2 × the sum of their covalent radii (`structures.BOND_SCALE`); the
  drawn bonds use Settings' bond tolerance (1.15 by default, which bonds the
  same pairs in every example structure). Very large cells (more than about
  4000 displayed atoms) are drawn without bonds.
- Setup previews ask the CLI each time the settings change (about 0.3-1.5 s
  locally, cached for settings it has seen). Run Preflight checks with
  CRYSTAL's TESTPDIM where a serial `crystal` is found (on HPCC through the
  CRYSTAL module); elsewhere it runs the structure check only and says so.
- FREQ normal modes are animated by the GUI itself, from the output's FREQ
  section. The HTML pages `mace plotting --freq` writes load plotly from a CDN
  and stay blank on nodes without internet access; the GUI needs neither.
- MACE drives CRYSTAL only. Export Structure lists a Quantum ESPRESSO input
  but disables it ("Needs the Quantum ESPRESSO backend"); there is no Quantum
  ESPRESSO support yet.
- The page follows the light and dark themes and reduced motion, and its
  colors meet WCAG 2.1 AA contrast; a high-contrast theme
  (`prefers-contrast: more`) is not there yet.

## Troubleshooting

### Reporting a problem

**Help › Copy Diagnostics** copies a report for a bug: MACE and Python
versions, the OS, the browser and its WebGL, how the server runs, package
versions, recent errors, failed requests and the last server log lines.
Paths, file and folder names, free-text settings, tokens, user and host names
are removed; read it before you share it. **Help › Report a Problem…** shows
the same text to edit, copy or save (`mace-gui-diagnostics.txt`), with a link
to open an issue on GitHub. When the GUI does not start at all, `mace gui
diagnostics --dir <folder>` prints the server's part.

### mace gui connect: key passphrases and the gateway

- **"ssh could not log in with your key"**: ssh ended with "Permission denied
  (publickey)" or could not unlock a key. When your key has a passphrase,
  ssh may have no terminal to ask for it on (often so on Windows, or when run
  from a script). Load the key into an agent
  once per login session, then run `mace gui connect` again:
  `eval "$(ssh-agent)"` and `ssh-add` (Linux, macOS, MobaXterm), or on
  Windows start the **OpenSSH Authentication Agent** service (in Services,
  set its startup type to Manual and start it) and run `ssh-add`. A Host
  alias from `~/.ssh/config` (with its `HostName`, `User` and
  `IdentityFile`) works as the gateway too: `--gateway hpcc`.
- **"the gateway … looks like an email address"**: the gateway is the login
  host you `ssh` to, `user@login-host` (e.g. `netid@login.example.edu`) or a
  Host alias from `~/.ssh/config`, not your email address
  (`netid@university.edu`). The warning shows when the part after `@` is a
  bare `name.tld` and no `Host` pattern in `~/.ssh/config` matches it; a real
  login host of that form gets the warning too, which a Host alias for it
  silences. MACE still connects. Fix the saved profile (in
  `~/.config/mace/connect.toml`, on Windows `%APPDATA%\mace\connect.toml`)
  with `mace gui connect [PROFILE] --gateway netid@login.example.edu --save`.
  The standalone scripts warn the same way.

### Messages

- **"3D view unavailable" where the structure should be:** this browser cannot
  draw 3D here (WebGL is off or not supported, which is common in remote
  desktops). Plots, text, the database and Setup still work; Export Image (PNG
  and SVG) and the tools that pick atoms in 3D say "Needs 3D (WebGL)". Try
  another browser, turn on hardware acceleration in the browser's settings,
  or open MACE GUI from your own computer through an ssh tunnel. The status
  bar shows "No 3D" while this is so.
- **"cannot listen on 127.0.0.1:8765 … Address already in use"** (on the
  node): someone else, or another `mace gui` of yours, uses that port. Pick
  your own (`--port 8xxx`, or `export MACE_GUI_PORT=8xxx`); `--port 0` picks a
  free one for a one-off run. (`mace gui connect` picks a random port on the
  node itself and tries again.)
- **`bind: Address already in use`** (from `ssh -L` on your computer): the
  port is taken there. Change the *first* number of the `-L` argument
  (`-L 9000:127.0.0.1:8xxx`) and use that port in the link (`mace gui
  connect` takes the next free one itself).
- **"This sign-in link was already used or has expired."**: each link works
  once (also after a link preview opened it). Run `mace gui login-url`, press
  Enter in `mace gui connect`, or paste the token into the form.
- **A page titled MACE that asks to sign in (403):** no session, or the server
  restarted with a new token. Open the link `mace gui` printed, or sign in
  with the token. In VS Code, the Ports tab opens this page: Ctrl+click the
  link in the terminal instead.
- **Signed Out:** the session was not used for the time set in Settings ›
  Server. Paste the token to continue; drafts and tasks are kept.
- **404 at `http://127.0.0.1:<port>/`:** the URL lacks its `/s/…/` session
  path. Open the link that `mace gui` printed.
- **421 "Misdirected request":** the page was opened by the node's name or
  real address. Use `http://127.0.0.1:<port>/…` through the tunnel (or
  `--allow-host`, with `--insecure-network`, if you really serve it on the
  network).
- **Wrong node:** a tunnel to another dev node behind the same gateway gives
  "connection refused" or "channel … open failed". `mace gui` names the node
  it runs on; the tunnel (or `--node`) must go to that exact node.
- **2FA:** `mace gui connect` waits for your approval and says "(waiting:
  approve the 2FA prompt if there is one)" after 15 s. With `--hop jump` a node
  that asks for 2FA too means a second prompt.
- **"mace: command not found"** (from `mace gui connect`): the node's login
  shell does not find `mace`. Pass `--mace <full path to mace_cli>` or a
  `--setup` line that puts it on PATH.
- **"The MACE on <node> is older than this connect"**: update MACE on the
  node, or use the manual tunnel.
- **"The node did not accept a login through the gateway"**: the node accepts
  logins only from the login node; `--hop nested` (auto does it once).
- **"… administratively prohibited"** (from ssh): your cluster turns off port
  forwarding on that host; ask your administrators.
- **A server still running on the node** after a dropped connection: it stays
  only while GUI tasks run, then stops by itself; `mace gui connect` again
  reattaches to it. As a last resort, `pkill -u $USER -f 'mace gui --managed'`
  on the node.
- **"stopped before this node's CPU-time limit"**: see
  [Long sessions on shared nodes](#long-sessions-on-shared-nodes).
- **A panel says it needs a package (503 with a `pip install` line):** run
  `pip install -r requirements.txt` for the Python that runs `mace` (on a
  cluster after the same `module load` lines).
- **A blank or gray 3D view:** WebGL2 is off in the browser, often in a remote
  desktop or a virtual machine. Check `about:support` (Firefox) or
  `chrome://gpu` (Chrome, Edge) for WebGL2.
- **"Lost the connection to the MACE server":** the ssh tunnel closed or the
  server stopped (the end of an `salloc`, Ctrl+C). Your edits are kept in the
  tab and saved as drafts; start the tunnel or the server again and open the
  link it prints.
- **Leftovers after a crash:** see the table in [Security](#security): the
  `$TMPDIR/mace_gui_cache_*` folder and the state file are safe to delete.

## For developers

### Measuring the editor's speed on a GPU

The default bench draws in software, so its orbit frame times are only
information. On a workstation with a second GPU, run the orbit headful on
that GPU once:

```bash
MACE_GUI_BENCH=1 MACE_GUI_BENCH_GPU1=1 nice -n 10 pytest -m guibench \
    tests/gui/bench/test_bench_editor.py -k orbit
```

It starts Firefox through PRIME offload on GPU1 (never GPU0, which drives the
display), checks that GPU1 has at least 1 GB free first, and fails if any of
its processes appears on GPU0.

### Tests

- `tests/test_gui_examples.py`: the server, its token and access rules, the
  readers, the before/after comparison and the GUI → JSON → CLI loop, through
  `mace_cli gui` in a subprocess.
- `tests/test_gui_editor.py`, `tests/test_gui_clean.py`: every edit operation,
  symmetry-aware edits, undo/redo, the CIF → cif2d12 round trip, the editor API,
  and Clean on distorted textbook cells.
- `tests/test_gui_results.py`, `tests/test_ipdos_readers.py`: the band/DOS
  arrays on synthetic files, and a check that lifting ipDOS's readers into
  functions left its plot byte-identical.
- `tests/test_gui_viewer.py`: formulas, element colors, slabs, and
  `node --check` on the page's own scripts.
- `tests/test_gui_browser.py`: the Structure, Results and Editor tabs in
  headless Firefox (selenium; skipped without Firefox or WebGL). With
  `MACE_GUI_SHOTS=<dir>` it also saves light and dark screenshots.
- `tests/test_american_english.py` (the GUI's front end, these docs, the
  logo and the example folder) and `tests/test_gui_american_english.py` (the
  example database): American English.
- `tests/test_gui_smoke_cli.py`, `tests/test_gui_writes.py`,
  `tests/test_gui_help.py`, `tests/test_gui_cli_startup.py`: the release
  gates through `mace_cli gui` (access rules, the session path and cookie
  scope, the token rules and the URL file, connection limits, SIGTERM and a
  hangup (also of a real terminal), writes that refuse symbolic links, the help, and that plain commands import no GUI
  code). `tests/gui/test_child_env.py` and `tests/gui/test_pool_internals.py`:
  the Python children's environment and the pool (no orphan after SIGTERM,
  Ctrl+C, a hangup or SIGKILL; a deck still written with one worker). `tests/gui/test_a11y_e2e.py` (axe-core, keyboard,
  reduced motion; needs `requirements-gui-test.txt`) and
  `tests/gui/test_tokens_contrast.py`: accessibility.
- `tests/gui/test_shell_e2e.py`: the window in headless Firefox: the menus
  against the command list (light and dark, 1280×800 and 1920×1080), every
  toolbar tooltip, the status bar, documents and the timeline, shortcuts,
  the command search, undo/redo and the modified marks, settings that survive
  a reload, themed scrollbars, unclipped labels and the same layout in both
  themes. Phase 2 adds the default keymap against the spec, user keymaps and
  the refused, risky and page-leaving keys, the menu mnemonics and F10, the
  palette's modes and speed, the Settings dialog (live viewport changes, the
  exact contents of gui.json, search, errors, the smallest window), the
  keymap editor, the shortcut sheet's columns and one-page print, a rebinding
  shown everywhere, the context menus, the leave prompt, and contrast and
  clipping in the dialogs. `tests/gui/test_shell_api.py` covers the settings
  file and its per-key checks, the resource limits, the CLI defaults, the
  event stream, the cached assets and the access rules of the new routes.
- `tests/gui/test_editor_e2e.py` and `tests/gui/test_editor_api.py`: the
  editing of Phase 3 in headless Firefox with real pointer actions (picking,
  box and lasso selection, the camera, every tool and its modal keys, Sym and
  P1, the inspector, the dialogs, measurements, the Atoms and History panels,
  saving with its confirmations and `.bak`, drafts across a reload, export at
  4× with transparency, New Structure and Edit This Step), and the server's
  side through the real `mace gui`: revisions, jumps, the one save path and
  the bounded `/api/symmetry`. `tests/gui/test_geom_core.py` runs the pure
  geometry module under node against numpy.
- `tests/gui/bench/` (opt-in, `MACE_GUI_BENCH=1`): the performance budgets
  (start time, first paint, the size of the page, tab switching, idle CPU,
  reload size, the heartbeat; and for Phase 2 the menu, palette and context
  menu open times, the palette filter on 500 commands and 2000 atoms, the
  tooltip, and that Settings and the shortcut sheet load on first open; for
  Phase 3 the worker's bond finding at 10k atoms, box selection at 8000, edit
  to paint at 1000, the symmetry requests during a drag and the server's
  threads and CPU meanwhile, ordered nudges and undos, the virtualized Atoms
  panel, hidden and freed viewers, and the server's memory with the cache
  filled; for Phase 4, `test_bench_files.py`: the batch file details on 2050
  files and their speed-up over one worker, Clean in the pool and four at
  once, cancel latencies, the parse cache, packed payload size, the 34 MB
  output's summary and step, the pool's workers, nice, threads, memory and
  idle shutdown, the mixed CPU load against the cap, and with
  `MACE_GUI_BENCH_BIG=1` the 85k-atom supercell) and
  the resources used: CPU workers, peak
  memory and the memory added on each GPU. It writes `bench_report.json`, into
  `$MACE_GUI_BENCH_OUT` if set; `MACE_GUI_BENCH_TUNNEL=<url>` repeats the
  browser rows over an ssh tunnel. Run it on its own and niced
  (`nice -n 10 pytest -m guibench tests/gui/bench`), never with `pytest -n`.
- Phase 4: `tests/gui/test_tasks_*.py` (the pool and tasks), `test_files_*.py`
  (cache, held documents, export, the banner, prefetch, the trajectory
  summary), `test_parse_api.py` and `test_drop_paste_e2e.py` (drop, import,
  paste), `test_recent_e2e.py`, `test_palette_rank_e2e.py`, `test_outindex.py`
  (the index against `read_out_frames` on every example output and a generated
  34 MB run), `test_builders_*.py`, `test_preview_lifecycle.py`,
  `test_builder_entry_e2e.py`, and `test_builder_decks.py`, which builds the
  same supercells, slab and vacancy with ase alone, converts them with the real
  `mace convert`, and checks the GUI's decks against them line for line.
- Phase 5: `test_setup_api.py`, `test_setup_routes.py` and `test_setup_e2e.py`
  (Setup's fields against the CLI's questions, and the matrix of calculation
  types where the GUI's saved settings give the manual, interactive route's
  deck byte for byte), `test_cli_tasks.py` (CLI runs as niced, cancelable
  process groups), `test_text_window.py`, `test_db_api.py` and
  `test_db_e2e.py` (a generated 10k-material database), `test_plot_e2e.py`,
  `test_freq_api.py` and `test_freq_e2e.py` (IR, Raman and FREQ modes on
  copies of `test/FREQ`, skipped without the corpus), `test_compare_api.py`
  and `test_compare_e2e.py` (100 shuffled MgO supercells matched exactly, the
  four looks, the dark ghost's contrast), `test_coordination_e2e.py`,
  `test_svg_export_e2e.py`, and `test_p5_route_security.py` (the token, Host,
  header and path rules of every new route, with a check that fails on a
  route it does not list). `bench/test_bench_analysis.py` measures the
  database, text, plotly, Setup, FREQ, polyhedra and SVG rows, and with
  `MACE_GUI_BENCH_TUNNEL` the same through ssh to an HPCC dev node, including
  Run Preflight with the serial `crystal` there.
- **Interpreters.** The GUI tests are run on Python 3.12 and 3.11, with ase,
  spglib, plotly and selenium. CI installs only `requirements-test.txt`, so
  the browser tests and those needing ase skip there; the browser tests need
  `pip install -r requirements-gui-test.txt` and Firefox.
- The headless Firefox of these tests draws WebGL in software (Mesa's
  llvmpipe), so they never use a GPU; a test checks the renderer.

import numpy as np
import matplotlib as mpl

mask_cmap = mpl.colors.ListedColormap(['white', 'grey', 'grey', 'grey'])

Bighand2RHD_skeidx = [0, 8, 7, 6, 1, 11, 10, 9, 2, 14, 13, 12, 3, 17, 16, 15, 4, 20, 19, 18, 5]
RHD2FreiHand_skeidx = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]

colorlist_pred = ['#660000', '#b30000', '#ff0000', '#ff4d4d', '#ff9999']
colorlist_gt = ['#000066', '#0000b3', '#0000ff', '#4d4dff', '#9999ff']

def plot_pose2d(axis, coords_hw, vis=None, color_fixed=None, linewidth='1', order='uv', draw_kp=True, markersize = 15):
    """
    Plots a hand stick figure into a matplotlib figure. revised based on Freihand
    input idx: Bighand
    hand idx: Freihand
    example:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(hand_image,alpha=0.6)
        plot_pose2d(ax1, pose2d, order='uv', draw_kp=True, linewidth='3',color_fixed='dimgray')
        ax1.axis('off')
        plt.show()
        plt.close()
    """
    coords_hw = coords_hw[Bighand2RHD_skeidx,:][RHD2FreiHand_skeidx,:]
    if order == 'uv':
        coords_hw = coords_hw[:, ::-1]

    colors = np.array([[0.4, 0.4, 0.4],
                       [0.4, 0.0, 0.0],
                       [0.6, 0.0, 0.0],
                       [0.8, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.4, 0.4, 0.0],
                       [0.6, 0.6, 0.0],
                       [0.8, 0.8, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.0, 0.6, 0.3],
                       [0.0, 0.8, 0.4],
                       [0.0, 1.0, 0.5],
                       [0.0, 0.2, 0.4],
                       [0.0, 0.3, 0.6],
                       [0.0, 0.4, 0.8],
                       [0.0, 0.5, 1.0],
                       [0.4, 0.0, 0.4],
                       [0.6, 0.0, 0.6],
                       [0.7, 0.0, 0.8],
                       [1.0, 0.0, 1.0]])

    colors = colors[:, ::-1]

    # define connections and colors of the bones
    bones = [((0, 1), colors[1, :]),
             ((1, 2), colors[2, :]),
             ((2, 3), colors[3, :]),
             ((3, 4), colors[4, :]),

             ((0, 5), colors[5, :]),
             ((5, 6), colors[6, :]),
             ((6, 7), colors[7, :]),
             ((7, 8), colors[8, :]),

             ((0, 9), colors[9, :]),
             ((9, 10), colors[10, :]),
             ((10, 11), colors[11, :]),
             ((11, 12), colors[12, :]),

             ((0, 13), colors[13, :]),
             ((13, 14), colors[14, :]),
             ((14, 15), colors[15, :]),
             ((15, 16), colors[16, :]),

             ((0, 17), colors[17, :]),
             ((17, 18), colors[18, :]),
             ((18, 19), colors[19, :]),
             ((19, 20), colors[20, :])]

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])

        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth, alpha = 0.6, zorder=1)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth, zorder=1)

    if not draw_kp:
        return

    for i in range(21):
        if vis[i] > 0.5:
            if color_fixed is None:
                axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=colors[i, :], markersize = markersize, zorder=1)
            else:
                axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=color_fixed, markersize = markersize, zorder=1)

    axis.set_zorder(0)

def plot_pose3d(points, plt_specs, ax, c = colorlist_gt, azim=-90.0, elev=180.0, grid=False):
    """
    revised based on Cross VAE Hand
    input idx: bighand
    hand idx: RHD
    set azim to 0 or 45 to get other view
    example:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        plot_pose3d(pose3d, '.-', colorlist_gt, ax1, azim=90.0, elev=180.0)
        plt.show()
        plt.close()

    Note that we switch the y axis and z axis (ax.plot(to_plot[:, 0], to_plot[:, 2], to_plot[:, 1])
    because we need the rotation along with y axis
    """
    assert points.size == 21 * 3, "pose3d should have 63 entries, it has %d instead" % points.size
    points = points[Bighand2RHD_skeidx,:]
    for i in range(5):
        start, end = i * 4 + 1, (i + 1) * 4 + 1
        to_plot = np.concatenate((points[start:end], points[0:1]), axis=0)
        ax.plot(to_plot[:, 0], to_plot[:, 2], to_plot[:, 1], plt_specs, color=c[i])

    ax.view_init(azim=azim, elev=elev)

    RADIUS = 0.12  # space around the subject
    xroot, yroot, zroot = points[12, 0], points[12, 2], points[12, 1]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.dist = 7.5

    if grid:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.grid(True)
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_axis_off()

        # Get rid of the ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

        # Get rid of the panes (actually, make them white)
        white = (1.0, 1.0, 1.0, 0.0)
        ax.w_xaxis.set_pane_color(white)
        ax.w_yaxis.set_pane_color(white)
        # Keep z pane

        # Get rid of the lines in 3d
        ax.w_xaxis.line.set_color(white)
        ax.w_yaxis.line.set_color(white)
        ax.w_zaxis.line.set_color(white)

def export_pose3d_gif(points, file = 'rotation.gif', plt_specs = '.-', c = colorlist_gt, azim=90.0, elev=180.0, grid=False):
    from matplotlib import animation
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_pose3d(points, plt_specs, c, ax, azim, elev, grid)
    def rotate(angle):
        ax.view_init(azim=angle,elev=elev)
    rot_animation = animation.FuncAnimation(fig, rotate, frames=(azim+np.arange(0, 362, 10))%360, interval=100)
    rot_animation.save(file, dpi=80, writer='imagemagick')
    plt.close()

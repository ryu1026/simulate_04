import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Simulate:
    def __init__(self,
                 num_beads=1,
                 beads_diameter=10,
                 spot_diameter=1,
                 threshold=0.5,
                 max_iteration=100,
                 col_grid_size=100,
                 row_grid_size=100,
                 grid_step=0.5,
                 triangle_radius=5,
                 walk_step_size=10,
                 intensity_max=1,
                 intensity_min=1e-6,
                 photomul_gain=1e6,
                 photomul_dark_current=0,
                 photomul_quantum_efficiency=0.4,
                 amp_gain=200,
                 amp_band_width=10e6,
                 amp_noise_factor=2,
                 num_frames=100,
                 do_print=True,
                 do_csv=True) -> object:

        self.num_beads = num_beads
        self.beads_diameter = beads_diameter
        self.spot_diameter = spot_diameter
        self.spot_radius = self.spot_diameter / 2
        self.threshold = threshold
        self.max_iteration = max_iteration
        self.col_grid_size = col_grid_size
        self.row_grid_size = row_grid_size
        self.grid_step = grid_step
        self.walk_step_size = walk_step_size
        self.triangle_radius = triangle_radius
        self.x0 = self.col_grid_size // 2
        self.y0 = self.row_grid_size // 2
        self.intensity_max = intensity_max
        self.intensity_min = intensity_min
        self.photomul_gain = photomul_gain
        self.photomul_dark_current = photomul_dark_current
        self.photomul_quantum_efficiency = photomul_quantum_efficiency
        self.amp_gain = amp_gain
        self.amp_band_width = amp_band_width
        self.amp_noise_factor = amp_noise_factor
        self.num_frames = num_frames
        self.do_print = do_print
        self.do_csv = do_csv

        # コンストラクタで格子点を作成していることに注意
        self.grid_row, self.grid_col = np.meshgrid(np.arange(0, self.col_grid_size, self.grid_step),
                                                   np.arange(0, self.row_grid_size, self.grid_step))

        # 蛍光ビーズを作成
        beads_col = []
        beads_row = []
        # 蛍光ビーズの中心座標をランダムに決定
        # 行方向の位置を決定
        # 蛍光ビーズが視野の端につくられないように範囲指定
        while len(beads_col) < self.num_beads:
            bead_col = np.random.randint(self.beads_diameter, self.col_grid_size - self.beads_diameter)
            bead_row = self.row_grid_size - np.random.randint(self.beads_diameter,
                                                              self.row_grid_size - self.beads_diameter)


            # 作成した点が蛍光ビーズの直径よりも大きい必要がある
            too_close = False
            for pre_col, pre_row in zip(beads_col, beads_row):
                # 蛍光ビーズの直径よりも離れた空間点で閾値を超える蛍光信号が検出された場合にはランダムウォーク，3点計測を行う
                if np.sqrt((bead_col - pre_col) ** 2 + (bead_row - pre_row) ** 2) < 2 * self.beads_diameter:
                    too_close = True
                    break

            if not too_close:
                beads_col.append(bead_col)
                beads_row.append(bead_row)

        mask = np.zeros_like(self.grid_row)
        # clipped_mask = np.zeros_like(self.grid_col)
        for bead_col, bead_row in zip(beads_col, beads_row):
            if self.do_print:
                # print("beads_pos: (行, 列) = ({0}, {1})".format(bead_col, abs(self.row_grid_size - bead_row)))
                print("beads_pos: (行, 列) = ({0}, {1}) = (y, x)".format(bead_col, bead_row))
            # ランダムに決定した中心と格子点の距離を計算
            dist = np.sqrt((self.grid_col - bead_col) ** 2 + (self.grid_row - bead_row) ** 2)
            # 現在のビーズの中心からself.beads_diameter/2 よりも近い全てのグリッド点で1に更新される
            mask += dist < (self.beads_diameter / 2)
        self.beads_matrix = np.clip(mask, 0., 1.)

        # 蛍光ビーズの中心の初期位置をcsvファイルに記録
        data = {'beads_col_center': beads_col, 'beads_row_center': beads_row}
        df = pd.DataFrame(data)
        filename = 'initial_beads_pos.xlsx'
        df.to_excel(filename, index=False)

    def make_grid_and_gaussian_beads(self):
        """
        グリッド (self.x_grid_size×self.y_grid_size) とガウス関数で近似した蛍光ビーズを作成する
        蛍光ビーズはnp.clip(beads, 0, 1)で0から1に範囲指定を行っていることに注意

        Return:
        -------
        numpy.ndarray
            グリッド上に配置された蛍光ビーズ
        """

        # gird_xは列方向に同じ値のベクトルを作成 ->
        # [[0, 0.1, 0.2]
        #  [0, 0.1, 0.2]]
        # gird_yは行方向に同じ値のベクトルを作成
        # [[0,     0,   0]
        #  [0.1, 0.1, 0.1]]
        # 作成された格子点 左上が原点のx-y二次元空間のイメージ
        # xは右に行くほど，yは下に行くほど値が増大する
        # (0,   0) (0.1,   0) (0.2,   0)
        # (0, 0.1) (0.1, 0.1) (0.2, 0.1)
        # (0. 0.2) (0.1, 0.2) (0.2, 0.2)
        grid_x, grid_y = np.meshgrid(np.arange(0, self.x_grid_size, self.grid_step),
                                     np.arange(0, self.y_grid_size, self.grid_step))

        # 0~grid_sizeの範囲でビーズの個数分=num_beads個の乱数を出す
        # 蛍光ビーズの座標をランダムに生成
        beads_x = np.random.uniform(0, self.x_grid_size, self.num_beads)
        beads_y = np.random.uniform(0, self.y_grid_size, self.num_beads)

        beads = np.zeros_like(grid_x)
        clip_beads = np.zeros_like(grid_x)
        # 各ビーズについて，ビーズの座標を中心とした2次元ガウス関数で強度を返す
        # ガウス関数は，グリッド上の各点におけるビーズの蛍光強度を示す．
        for bead_x, bead_y in zip(beads_x, beads_y):
            dist = np.sqrt((grid_x - bead_x) ** 2 + (grid_y - bead_y) ** 2)
            bead_z = np.exp(-dist ** 2 / (2 * (self.beads_diameter / 2) ** 2))
            # bead_z = np.exp(-((grid_x - bead_x)**2 + (grid_y - bead_y)**2) / (2 * (self.beads_diameter/2)**2))
            beads += bead_z
            clip_beads = np.clip(beads, 0, 1)

        return beads, clip_beads

    def make_not_gaussian_beads(self, do_print=False):
        """
        ガウシアンでない蛍光ビーズを作成する
        ビーズ内の蛍光体の分布は一様である事に注意

        :param do_print: bool 蛍光ビーズの座標を表示するかどうか (Excelに書き出す?)
        :return beads_matrix: numpy.ndarray (x_grid_size, y_grid_size)で0か1の値が乗っている
        """

        # 蛍光ビーズの中心座標をランダムに決定
        # 行方向の位置を決定
        # 蛍光ビーズが視野の端につくられないように範囲指定
        beads_col = np.random.randint(self.beads_diameter, self.col_grid_size-self.beads_diameter, self.num_beads)
        # 列方向の位置を決定
        # beads_row = np.random.randint(0, self.row_grid_size, self.num_beads)
        beads_row = np.random.randint(self.beads_diameter, self.row_grid_size-self.beads_diameter,
                                      self.num_beads)
        mask = np.zeros_like(self.grid_row)
        # clipped_mask = np.zeros_like(self.grid_col)
        print("beads_col", beads_col)
        for bead_col, bead_row in zip(beads_col, beads_row):
            if do_print:
                # print("beads_pos: (行, 列) = ({0}, {1})".format(bead_col, abs(self.row_grid_size - bead_row)))
                print("beads_pos: (行, 列) = ({0}, {1}) = (y, x)".format(bead_col, bead_row))
            # ランダムに決定した中心と格子点の距離を計算
            dist = np.sqrt((self.grid_col - bead_col) ** 2 + (self.grid_row - bead_row) ** 2)
            # 現在のビーズの中心からself.beads_diameter/2 よりも近い全てのグリッド点で1に更新される
            mask += dist < (self.beads_diameter / 2)
            beads_matrix = np.clip(mask, 0., 1.)

        return beads_matrix, beads_col, beads_row

    def make_not_gaussian_beads_not_overlap(self, do_print=False):
        """
        ガウシアンでない蛍光ビーズを作成する
        このメソッドは蛍光ビーズ同士がオーバーラップしないように作成している
        ビーズ内の蛍光体の分布は一様である事に注意

        :param do_print: bool 蛍光ビーズの座標を表示するかどうか (Excelに書き出す?)
        :return beads_matrix: numpy.ndarray (x_grid_size, y_grid_size)で0か1の値が乗っている
        """
        beads_col = []
        beads_row = []
        # 蛍光ビーズの中心座標をランダムに決定
        # 行方向の位置を決定
        # 蛍光ビーズが視野の端につくられないように範囲指定
        while len(beads_col) < self.num_beads:
            bead_col = np.random.randint(self.beads_diameter, self.col_grid_size - self.beads_diameter)
            bead_row = self.row_grid_size - np.random.randint(self.beads_diameter, self.row_grid_size-self.beads_diameter)

            # 作成した点が蛍光ビーズの直径よりも大きい必要がある
            too_close = False
            for pre_col, pre_row in zip(beads_col, beads_row):
                # 蛍光ビーズの直径よりも離れた空間点で閾値を超える蛍光信号が検出された場合にはランダムウォーク，3点計測を行う
                if np.sqrt((bead_col - pre_col)**2 + (bead_row - pre_row)**2) < 2 * self.beads_diameter:
                    too_close = True
                    break

            if not too_close:
                beads_col.append(bead_col)
                beads_row.append(bead_row)

        mask = np.zeros_like(self.grid_row)
        # clipped_mask = np.zeros_like(self.grid_col)

        for bead_col, bead_row in zip(beads_col, beads_row):
            if do_print:
                # print("beads_pos: (行, 列) = ({0}, {1})".format(bead_col, abs(self.row_grid_size - bead_row)))
                print("beads_pos: (行, 列) = ({0}, {1}) = (y, x)".format(bead_col, bead_row))
            # ランダムに決定した中心と格子点の距離を計算
            dist = np.sqrt((self.grid_col - bead_col) ** 2 + (self.grid_row - bead_row) ** 2)
            # 現在のビーズの中心からself.beads_diameter/2 よりも近い全てのグリッド点で1に更新される
            mask += dist < (self.beads_diameter / 2)
        beads_matrix = np.clip(mask, 0., 1.)

        return beads_matrix, beads_col, beads_row

    def make_not_gaussian_beads_not_random(self, x_pos, y_pos, do_print=False):
        """
        ガウシアンでない蛍光ビーズを作成する
        ビーズ内の蛍光体の分布は一様である事に注意

        :param do_print: bool 蛍光ビーズの座標を表示するかどうか (Excelに書き出す?)
        :return beads_matrix: numpy.ndarray (x_grid_size, y_grid_size)で0か1の値が乗っている
        """

        # 蛍光ビーズの中心座標をランダムに決定
        beads_x = x_pos
        beads_y = y_pos

        mask = np.zeros_like(self.grid_x)
        clipped_mask = np.zeros_like(self.grid_x)

        for bead_x, bead_y in zip(beads_x, beads_y):
            if do_print:
                print("beads_pos: (x, y) = ({0}, {1})".format(bead_x, abs(self.y_grid_size - bead_y)))

            # ランダムに決定した中心と格子点の距離を計算
            dist = np.sqrt((self.grid_x - bead_x) ** 2 + (self.grid_y - bead_y) ** 2)
            # 現在のビーズの中心からself.beads_diameter/2 よりも近い全てのグリッド点で1に更新される
            mask += dist < (self.beads_diameter / 2)
            beads_matrix = np.clip(mask, 0., 1.)

        return beads_matrix

    def draw_not_gaussian_beads(self, not_gaussian_beads) -> object:

        fig, ax = plt.subplots()
        ax.imshow(not_gaussian_beads, cmap='gray', interpolation='nearest',
                  extent=(0, self.col_grid_size, 0, self.row_grid_size))
        ax.set_xlabel("x [µm]", fontname='MS Gothic', fontsize=18)
        ax.set_ylabel("y [µm]", fontname='MS Gothic', fontsize=18)
        # ax.set_title("蛍光ビーズ",fontname='MS Gothic', fontsize=20)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        # ax.invert_yaxis()
        # ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        # ax.axis('off')
        plt.tick_params(labelsize=14)
        plt.tight_layout()
        plt.show()

    def draw_not_gaussian_beads_invert_y(self, not_gaussian_beads):

        fig, ax = plt.subplots()
        ax.imshow(not_gaussian_beads, cmap='gray', interpolation='nearest',
                  extent=(0, self.col_grid_size, 0, self.row_grid_size))
        ax.set_xlabel("row (microns)")
        ax.set_ylabel("col (microns)")
        ax.set_title("Not gaussian fluorescent beads")
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.set_yticks([])
        ax.set_yticklabels([])
        # ax.invert_yaxis()
        plt.show()

    def draw_fluo_beads(self, beads):
        fig, ax = plt.subplots()
        ax.imshow(beads, cmap='gray', extent=(0, self.x_grid_size, 0, self.y_grid_size))
        ax.set_xlabel('x (microns)')
        ax.set_ylabel('y (microns)')
        ax.set_title('Gaussian fluorescent beads')
        plt.show()

    def generate_spot(self, beam_pos_x, beam_pos_y, do_threshold=True, do_draw=False):
        """
        詳細はbeam_sim.pyを参照
        集光点の中心座標をもとにガウシアンビームを作成する
        :param beam_pos_x: 集光点のx座標
        :param beam_pos_y: 集光点のy座標
        :param do_threshold: bool ガウシアンビームの強度値が小さい領域は0でマスクするかどうか
        :param do_draw: bool 作成したスポットを描画する

        :return spot_intensity: 中心(x,y)でスポットの直径がself.spot_diameterのビーム（強度）
                 numpy.ndarray (1000×1000) <- 100×100の範囲でstep_sizeが0.1だから．
        :return [x_pos, y_pos]: list座標のリスト 本来は呼び出す側がリストを保持しておけばよいはず
        """
        # おそらくself.grid_stepで割る必要はない
        # sigma = self.spot_diameter / (2 * np.sqrt((2*np.log(2)))) / self.grid_step
        sigma = self.spot_diameter / (2 * np.sqrt((2 * np.log(2))))

        # グリッドと実際のy座標は増大方向が反対だから差の絶対値をとる
        # beam_pos_y_gridはグリッド空間上での座標 (グリッド空間ではyは下に行くほど大きい)
        # 実空間ではyは上に行くほど大きいからその分を補正
        beam_pos_y_grid = abs(self.y_grid_size - beam_pos_y)

        r = np.sqrt((self.grid_x - beam_pos_x) ** 2 + (self.grid_y - beam_pos_y_grid) ** 2)

        spot_intensity = self.intensity_max * np.exp(-r ** 2 / (2 * sigma ** 2))

        # ガウシアンビームだと範囲が広いのでself.intensity_min以下の強度の場合は0にする
        if do_threshold:
            spot_intensity[spot_intensity < self.intensity_min] = 0

        # 作成したスポットを確認するかどうか
        if do_draw:
            print("Spot pos: (x, y) = ({0}, {1})".format(beam_pos_x, beam_pos_y))
            plt.imshow(spot_intensity, cmap='hot', extent=(0, self.x_grid_size, 0, self.y_grid_size))
            plt.xlabel('x (µm)')
            plt.ylabel('y (µm)')
            plt.title('Gaussian beam at ({0}, {1})'.format(beam_pos_x, beam_pos_y))
            plt.colorbar()
            plt.show()

        return spot_intensity, [beam_pos_x, beam_pos_y]

    def get_signal(self, beads_matrix, col_pos, row_pos):
        """
        指定された座標を中心に直径self.spot_diameterのガウシアンで信号を取得する
        beads_matrixの画素値にガウシアンをかけて量子効率を乗算する
        とりあえず量子効率は0.9で指定
        :return: 実際の
        """
        # スライス時には整数で範囲を指定する必要がある．
        # 集光スポットをグリッドの間隔に合わせないといけない
        # 集光スポットが1 µmであり，これはグリッドの10個分に相当する
        # 以下のコードは集光スポットの半径がself.spot_radiusで格子点の間隔はgrid_stepだから
        # 半径のグリッド数はself.spot_radius / self.grid_stepで求められる
        quantum_efficiency = 1.
        spot_radius_scale = int(self.spot_radius / self.grid_step)

        print("spot_radius_scale: ", spot_radius_scale)
        signal_area_value = beads_matrix[col_pos * 10, row_pos * 10]
        print("signal_value", signal_area_value)
        signal_area = beads_matrix[col_pos * 10 - spot_radius_scale:col_pos * 10 + spot_radius_scale,
                      row_pos * 10 - spot_radius_scale:row_pos * 10 + spot_radius_scale]
        dist_2 = (np.arange(signal_area.shape[0]) - col_pos * 10) ** 2 + (
                np.arange(signal_area.shape[1]) - row_pos * 10) ** 2
        print("signal_area.max", signal_area.max())
        print("r^2", dist_2)

        # I(r) = I0 * exp ( -2r^2 / w^2 )
        # w = FWHM / sqrt(2 * ln2)
        # FWHMの値はとりあえず2 * spot_radius_scaleにする
        # FWHMはビームの直径であるのに対しwは1/e^2の半径分に相当していることに注意
        fwhm = 2 * spot_radius_scale
        w_2 = fwhm ** 2 / (2 * np.log(2))
        gaussian = np.exp((-2 * dist_2) / w_2)
        print("fwhm: {0}, w_2: {1}".format(fwhm, w_2))
        print("gaussian.max()", gaussian.shape)
        signal = quantum_efficiency * np.sum(signal_area * gaussian)
        return signal

    def get_signal_simple(self, beads_matrix, col_pos, row_pos, count):
        """
        指定された座標を中心に直径self.spot_diameterの範囲から信号を取得する
        :param count: 実際の照射位置
        :param row_pos: x
        :param col_pos:
        :param beads_matrix:
        :return: 実際の
        """
        # スライス時には整数で範囲を指定する必要がある．
        # 集光スポットをグリッドの間隔に合わせないといけない
        # 集光スポットが1 µmであり，これはグリッドの10個分に相当する
        # 集光スポットの半径がself.spot_radiusで格子点の間隔はgrid_stepだから
        # 半径のグリッド数spot_radius_scaleはself.spot_radius / self.grid_stepで求められる
        # print("######{0}回目######".format(count))
        spot_radius_scale = int(self.spot_radius / self.grid_step)
        # print("spot_radius_scale: ", spot_radius_scale)

        signal_area_value = beads_matrix[col_pos * 10, row_pos * 10]
        # print("照射位置({0}, {1}): {2}".format(col_pos, row_pos, signal_area_value))

        # signal_areaは集光点のサイズをbeads_matrixにあてはめた状態
        signal_area = beads_matrix[col_pos*10-spot_radius_scale:col_pos*10+spot_radius_scale,
                                   row_pos*10-spot_radius_scale:row_pos*10+spot_radius_scale]

        # print("signal_area.shape (10, 10)になるはず", signal_area.shape)
        # print("signal_area.max", signal_area.max())

        # signal_area (10×10)に内接する円の平均シグナルをとる
        grid_size = 10
        x, y = np.indices((grid_size, grid_size))
        circle_indices = np.where((x - spot_radius_scale)**2 + (y - spot_radius_scale)**2 <= spot_radius_scale**2)
        # print("signal_area.min", signal_area.min())
        signal = np.mean(signal_area[circle_indices])
        # print("signal: {0}".format(signal))
        # print("###############")
        return signal

    def calculate_error(self, beads_matrix, col_pos, row_pos):
        """
        3点計測の中心座標とビーズの中心との距離を求める関数
        指定した座標と実際のグリッド間の距離を考慮する必要がある

        :param beads_matrix:
        :param col_pos:
        :param row_pos:
        :return:
        """
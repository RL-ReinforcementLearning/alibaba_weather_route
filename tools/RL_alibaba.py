import os
import sys
import pandas as pd
import random
from matplotlib import pyplot as plt
from timeit import default_timer as timer
import numpy as np
from tools.Dyna_3D import Dyna_3D
from tools.Maze3D import Maze_3D
import multiprocessing
from tools.simpleSub import a_star_submission_3d, collect_csv_for_submission
import fnmatch
from tools.visualisation import plot_state_action_value


def load_a_star_precompute(cf):
    A_star_model_precompute_csv = []
    if cf.day_list[0] <= 5:
        for m in range(10):
            csv_file = os.path.join(cf.A_star_precompute_path, 'Train_A_star_search_3D_conservative_wall_wind_15_model_number_[%d].csv'%(m+1))
            A_star_model_precompute_csv.append(pd.read_csv(csv_file, names=['target', 'date', 'time', 'xid', 'yid']))
    else:
        for m in range(10):
            csv_file = os.path.join(cf.A_star_precompute_path, 'Test_A_star_search_3D_conservative_wall_wind_15_model_number_[%d].csv'%(m+1))
            A_star_model_precompute_csv.append(pd.read_csv(csv_file, names=['target', 'date', 'time', 'xid', 'yid']))
    return A_star_model_precompute_csv


def process_wind(cf, day):
    wind_real_day_hour_total = np.zeros(shape=(len(cf.model_number), cf.grid_world_shape[0], cf.grid_world_shape[1], int(cf.total_hours)))
    for hour in range(3, 21):
        if day < 6:  # meaning this is a training day
            if cf.use_real_weather:
                weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
            else:
                wind_real_day_hour_temp = []
                for model_number in cf.model_number:
                    # we average the result
                    weather_name = 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                    wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                    wind_real_day_hour_temp.append(wind_real_day_hour_model)
                wind_real_day_hour = np.asarray(wind_real_day_hour_temp)
        else:
            wind_real_day_hour_temp = []
            for model_number in cf.model_number:
                # we average the result
                weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                wind_real_day_hour_temp.append(wind_real_day_hour_model)
            wind_real_day_hour = np.asarray(wind_real_day_hour_temp)

        # we replicate the weather for the whole hour
        if cf.risky:
            wind_real_day_hour[wind_real_day_hour < cf.wall_wind] = 1  # Every movement will have a unit cost
        elif cf.wind_exp:
            wind_real_day_hour[
                wind_real_day_hour < cf.wall_wind] -= cf.wind_exp_mean  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
            wind_real_day_hour[
                wind_real_day_hour < cf.wall_wind] /= cf.wind_exp_std  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
            wind_real_day_hour[wind_real_day_hour < cf.wall_wind] = np.exp(
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind]).astype(
                'int')  # with int op. if will greatly enhance the computatinal speed
        else:
            wind_real_day_hour[
                wind_real_day_hour < cf.wall_wind] /= cf.risky_coeff  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
            wind_real_day_hour[wind_real_day_hour < cf.wall_wind] += 1

        wind_real_day_hour_total[:, :, :, hour - 3] = wind_real_day_hour[:, :, :]  # we replicate the hourly data

    return wind_real_day_hour_total


def extract_weather_data_model(cf, day, hour, model_number):
    if day < 6:  # meaning this is a training day
        weather_name = 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
        rainfall_name = 'Train_forecast_rainfall_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
    else:
        weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
        rainfall_name = 'Test_forecast_rainfall_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)

    wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
    wind_real_day_hour_model = np.round(wind_real_day_hour_model, 2)
    # rainfall
    rainfall_real_day_hour_model = np.load(os.path.join(cf.rainfall_save_path, rainfall_name))
    rainfall_real_day_hour_model = np.round(rainfall_real_day_hour_model, 2)

    return wind_real_day_hour_model, rainfall_real_day_hour_model


def process_wind_and_rainfall(cf, day, start_hour):
    # deal weather data
    wind_real_day_hour_total = np.zeros(shape=(11, cf.grid_world_shape[0], cf.grid_world_shape[1], cf.hour_unique[1] - start_hour + 1))
    rainfall_real_day_hour_total = np.zeros(shape=(11, cf.grid_world_shape[0], cf.grid_world_shape[1], cf.hour_unique[1] - start_hour + 1))
    for hour in range(start_hour, 21):
        # --------extract weather data ---------#
        for model_number in range(1, 11):
            wind_real_day_hour, rainfall_real_day_hour = extract_weather_data_model(cf, day, hour, model_number)
            wind_real_day_hour_total[model_number-1, :, :, hour - start_hour] = wind_real_day_hour
            rainfall_real_day_hour_total[model_number-1, :, :, hour - start_hour] = rainfall_real_day_hour

        # We also replicate the mean weather
        wind_real_day_hour_total[10, :, :, hour - start_hour] = np.mean(wind_real_day_hour_total[:10, :, :, hour - start_hour], axis=0)
        rainfall_real_day_hour_total[10, :, :, hour - start_hour] = np.mean(rainfall_real_day_hour_total[:10, :, :, hour - start_hour], axis=0)

    return wind_real_day_hour_total, rainfall_real_day_hour_total


def initialise_maze_and_model(cf, start_loc_3D, goal_loc_3D, wind_real_day_hour_total):
    maze = Maze_3D(height=cf.grid_world_shape[0],
                   width=cf.grid_world_shape[1],
                   time_length=int(cf.time_length),
                   start_state=start_loc_3D,
                   goal_states=goal_loc_3D,
                   reward_goal=cf.reward_goal,
                   reward_move=cf.reward_move,
                   maxSteps=cf.maxSteps,
                   wind_real_day_hour_total=wind_real_day_hour_total,
                   c_baseline=cf.c_baseline_a_star,
                   cf=cf)

    model = Dyna_3D(rand=np.random.RandomState(cf.random_state),
                    maze=maze,
                    epsilon=cf.epsilon,
                    gamma=cf.gamma,
                    planningSteps=cf.planningSteps,
                    qLearning=cf.qLearning,
                    expected=cf.expected,
                    alpha=cf.alpha,
                    priority=cf.priority,
                    theta=cf.theta,
                    optimal_length_relax=cf.optimal_length_relax,
                    polynomial_alpha=False,
                    heuristic=cf.heuristic)
    return model


def extract_a_star(A_star_model_precompute_csv, day, goal_city):
    go_to_all = []
    for csv in A_star_model_precompute_csv:
        go_to = {}
        predicted_df_now = csv.loc[(csv['date'] == day) & (csv['target'] == goal_city)]
        for t in range(len(predicted_df_now)-1):
            go_to[(predicted_df_now.iloc[t]['xid']-1, predicted_df_now.iloc[t]['yid']-1, t)] = \
                (predicted_df_now.iloc[t+1]['xid']-1, predicted_df_now.iloc[t+1]['yid']-1, t)
        go_to_all.append(go_to)

    return go_to_all


def draw_predicted_route(cf, day, go_to_all, city_data_df, route_list):
    plt.close("all")
    # draw figure maximum
    plt.figure(1)
    plt.clf()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    for hour in range(3, 21):
        if day < 6:  # meaning this is a training day
            weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
            wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
        else:
            # wind_real_day_hour_temp = []
            # for model_number in cf.model_number:
            #     # we average the result
            #     weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
            #     wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
            #     wind_real_day_hour_temp.append(wind_real_day_hour_model)
            # wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
            # wind_real_day_hour = np.mean(wind_real_day_hour_temp, axis=0)

            # we use model 3 here because it has the best result amongst all the models
            weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (3, day, hour)
            wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
            weather_name = 'Test_forecast_wind_model_count_%d_day_%d_hour_%d.npy' % (len(cf.model_number), day, hour)
        title = weather_name[:-4]
        plt.clf()
        plt.imshow(wind_real_day_hour, cmap=cf.colormap)
        plt.colorbar()

        # we also plot the city location
        for idx in range(city_data_df.index.__len__()):
            x_loc = int(city_data_df.iloc[idx]['xid']) - 1
            y_loc = int(city_data_df.iloc[idx]['yid']) - 1
            cid = int(city_data_df.iloc[idx]['cid'])
            plt.scatter(y_loc, x_loc, c='r', s=40)
            if wind_real_day_hour[x_loc, y_loc] >= cf.wall_wind:
                plt.annotate(str(cid), xy=(y_loc, x_loc), color='black', fontsize=20)
            else:
                plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)
        # we also draw some contours
        x = np.arange(0, wind_real_day_hour.shape[1], 1)
        y = np.arange(0, wind_real_day_hour.shape[0], 1)
        X, Y = np.meshgrid(x, y)
        CS = plt.contour(X, Y, wind_real_day_hour, (15,), colors='k')

        for p in route_list:
            plt.scatter(p[1], p[0], c='yellow', s=5)

        for h in range(3, hour + 1):
            for p in route_list[(h - 3) * 30:(h - 2) * 30]:
                plt.scatter(p[1], p[0], c=cf.colors[np.mod(h, 2)], s=10)
                if h >= hour and wind_real_day_hour[p[0], p[1]] >= cf.wall_wind:
                    title = weather_name[:-4] + '.  Crash at: ' + str(p) + ' in hour: %d' % (
                        p[2] / 30 + 3)
                    print(title)


        ########## Plot all a-star
        for m, go_to in enumerate(go_to_all):
            for p in go_to.keys():
                plt.scatter(p[1], p[0], c='black', s=1, alpha=0.5)
            # anno_point = int(len(go_to.keys()) * (1+m) / 10)
            anno_point = int(len(go_to.keys()) / 2)
            go_to_sorted = sorted(go_to)
            p = go_to_sorted[anno_point]
            # plt.annotate(str(m+1), xy=(p[1], p[0]), color='indigo', fontsize=20)

        plt.clabel(CS, inline=1, fontsize=10)
        plt.title(title)
        plt.waitforbuttonpress(0.01)
        if 30 * (hour - 2) > len(route_list):
            break


def print_predicted_route(cf, day, go_to_all, city_data_df, route_list):
    """
    We only print whether the route is successful instead of the slow drawing
    :param cf:
    :param day:
    :param go_to_all:
    :param city_data_df:
    :param route_list:
    :return:
    """
    for hour in range(3, 21):
        if day < 6:  # meaning this is a training day
            weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
            wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
        else:
            weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (3, day, hour)
            wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
            weather_name = 'Test_forecast_wind_model_count_%d_day_%d_hour_%d.npy' % (len(cf.model_number), day, hour)

        for h in range(3, hour + 1):
            for p in route_list[(h - 3) * 30:(h - 2) * 30]:
                if h >= hour and wind_real_day_hour[p[0], p[1]] >= cf.wall_wind:
                    title = weather_name[:-4] + '.  Crash at: ' + str(p) + ' in hour: %d' % (p[2] / 30 + 3)
                    print(title)

                    # we draw the crash time here
                    plt.close("all")
                    # draw figure maximum
                    plt.figure(1)
                    plt.clf()
                    mng = plt.get_current_fig_manager()
                    mng.resize(*mng.window.maxsize())
                    plt.imshow(wind_real_day_hour, cmap=cf.colormap)
                    plt.colorbar()
                    # we also plot the city location
                    for idx in range(city_data_df.index.__len__()):
                        x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                        y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                        cid = int(city_data_df.iloc[idx]['cid'])
                        plt.scatter(y_loc, x_loc, c='r', s=40)
                        if wind_real_day_hour[x_loc, y_loc] >= cf.wall_wind:
                            plt.annotate(str(cid), xy=(y_loc, x_loc), color='black', fontsize=20)
                        else:
                            plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)
                    # we also draw some contours
                    x = np.arange(0, wind_real_day_hour.shape[1], 1)
                    y = np.arange(0, wind_real_day_hour.shape[0], 1)
                    X, Y = np.meshgrid(x, y)
                    CS = plt.contour(X, Y, wind_real_day_hour, (15,), colors='k')
                    plt.clabel(CS, inline=1, fontsize=10)

                    for p in route_list:
                        plt.scatter(p[1], p[0], c='yellow', s=5)
                    ########## Plot all a-star
                    for m, go_to in enumerate(go_to_all):
                        for p in go_to.keys():
                            plt.scatter(p[1], p[0], c='black', s=1, alpha=0.5)
                        # anno_point = int(len(go_to.keys()) * (1+m) / 10)
                        anno_point = int(len(go_to.keys()) / 2)
                        go_to_sorted = sorted(go_to)
                        p = go_to_sorted[anno_point]
                        # plt.annotate(str(m+1), xy=(p[1], p[0]), color='indigo', fontsize=20)
                    plt.title(title)
                    plt.waitforbuttonpress()
                    return False

        if 30 * (hour - 2) > len(route_list):
            return True


def print_predicted_route_wind_and_rainfall(cf, day, goal_city, df):
    """
    We only print whether the route is successful instead of the slow drawing
    :param cf:
    :param day:
    :param go_to_all:
    :param city_data_df:
    :param route_list:
    :return:
    """
    from tools.visualisation import get_wind_rainfall
    cf.model_number = list(range(1, 11))
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    fig = plt.figure(num=1)
    fig.clf()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    crash_wind = False
    crash_rainfall = False
    starting_hour = int(df.iloc[0]['time'][:2])

    for step in range(1, len(df)):
        hour = int(df.iloc[step]['time'][:2])
        hour_prev_step = int(df.iloc[step-1]['time'][:2])
        if hour != hour_prev_step:
            real_weather, mean_weather, real_wind, real_rainfall = get_wind_rainfall(cf, day, hour_prev_step)
            # so we finish prevous hour, draw it!
            clim_max = real_weather.max()
            plt.clf()
            weather_names = ['date: %d, city: %d, hour: %d' % (day, goal_city, hour_prev_step) + '_real-weather',
                             'mean weather', 'real wind', 'real rainfall']

            for (plot_number, im) in zip(range(4), [real_weather, mean_weather, real_wind, real_rainfall]):
                plt.subplot(2, 2, plot_number+1)
                plt.imshow(im, cmap=cf.colormap)
                if plot_number < 2:
                    plt.clim(0, clim_max)
                plt.colorbar()
                # we also plot the city location
                for idx in range(city_data_df.index.__len__()):
                    x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                    y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                    cid = int(city_data_df.iloc[idx]['cid'])
                    plt.scatter(y_loc, x_loc, c='r', s=40)
                    plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=10)
                # we also draw some contours
                x = np.arange(0, im.shape[1], 1)
                y = np.arange(0, im.shape[0], 1)
                X, Y = np.meshgrid(x, y)
                if plot_number==3:
                    CS = plt.contour(X, Y, im, (4,), colors='white')
                    plt.clabel(CS, inline=1, fontsize=10)
                else:
                    CS = plt.contour(X, Y, im, (15,), colors='k')
                    plt.clabel(CS, inline=1, fontsize=10)
                plt.title(weather_names[plot_number])

                # we plot every hour
                crash_wind = False
                crash_rainfall = False
                for i in range(step):
                    if i <= step-30:
                        plt.scatter(df.iloc[i]['yid'], df.iloc[i]['xid'], c='orange', s=1)
                    else:
                        # check whether we have crashed!
                        if real_wind[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 15:
                            crash_wind = True
                        elif real_rainfall[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 4.0:
                            crash_rainfall = True
                        if not crash_wind or crash_rainfall:
                            plt.scatter(df.iloc[i]['yid'], df.iloc[i]['xid'], c='orange', s=1)
                        else:
                            plt.scatter(df.iloc[i]['yid'], df.iloc[i]['xid'], c='red', s=1)

            plt.waitforbuttonpress(0.1)
            crash_wind = False
            crash_rainfall = False
            for i in range(max(0, step-30), step):
                # check whether we have crashed!
                min = int(df.iloc[i]['time'][3:])
                if real_wind[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 15:
                    crash_wind = True
                    plt.suptitle('Crash on wind:  Day: %d, city: %d, starting hour: %d, hour: %d, min: %d, xid: %d, yid : %d with cost: %.2f, mean cost: %2.f'
                                 % (day, goal_city, starting_hour, hour_prev_step, min, df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, real_wind[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1], mean_weather[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1]))
                    plt.scatter(df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, c='r', s=10)
                elif real_rainfall[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 4.0:
                    crash_rainfall = True
                    plt.suptitle('Crash on rainfall: Day: %d, city: %d, starting hour: %d, hour: %d, min: %d, xid: %d, yid : %d with cost: %.2f, mean cost: %2.f'
                                 % (day, goal_city, starting_hour, hour_prev_step, min, df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, real_rainfall[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1], mean_weather[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1]))
                    plt.scatter(df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, c='r', s=10)

                if crash_wind or crash_rainfall:
                    break

        if crash_wind or crash_rainfall:
            plt.waitforbuttonpress(0)
            break

    # The following code it the copy of previous code apart from the loop
    # We still need to plot the last bit
    if not (crash_wind or crash_rainfall):
        hour_prev_step = int(df.iloc[-1]['time'][:2])
        # so we finish prevous hour, draw it!
        clim_max = real_weather.max()
        plt.clf()
        weather_names = ['date: %d, city: %d, hour: %d' % (day, goal_city, hour_prev_step) + '_real-weather', 'mean weather', 'real wind', 'real rainfall']
        real_weather, mean_weather, real_wind, real_rainfall = get_wind_rainfall(cf, day, hour_prev_step)

        for (plot_number, im) in zip(range(4), [real_weather, mean_weather, real_wind, real_rainfall]):
            plt.subplot(2, 2, plot_number + 1)
            plt.imshow(im, cmap=cf.colormap)
            if plot_number < 2:
                plt.clim(0, clim_max)
            plt.colorbar()
            # we also plot the city location
            for idx in range(city_data_df.index.__len__()):
                x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                cid = int(city_data_df.iloc[idx]['cid'])
                plt.scatter(y_loc, x_loc, c='r', s=40)
                plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=10)
            # we also draw some contours
            x = np.arange(0, im.shape[1], 1)
            y = np.arange(0, im.shape[0], 1)
            X, Y = np.meshgrid(x, y)
            if plot_number == 3:
                CS = plt.contour(X, Y, im, (4,), colors='white')
                plt.clabel(CS, inline=1, fontsize=10)
            else:
                CS = plt.contour(X, Y, im, (15,), colors='k')
                plt.clabel(CS, inline=1, fontsize=10)
            plt.title(weather_names[plot_number])

            # we plot every hour
            crash_wind = False
            crash_rainfall = False
            for i in range(len(df)):
                if i <= len(df) - 30:
                    plt.scatter(df.iloc[i]['yid'], df.iloc[i]['xid'], c='orange', s=1)
                else:
                    # check whether we have crashed!
                    if real_wind[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 15:
                        crash_wind = True
                    elif real_rainfall[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 4.0:
                        crash_rainfall = True
                    if not crash_wind or crash_rainfall:
                        plt.scatter(df.iloc[i]['yid'], df.iloc[i]['xid'], c='orange', s=1)
                    else:
                        plt.scatter(df.iloc[i]['yid'], df.iloc[i]['xid'], c='red', s=1)

        plt.waitforbuttonpress(1)
        crash_wind = False
        crash_rainfall = False
        for i in range(max(0, len(df) - 30), len(df)):
            # check whether we have crashed!
            min = int(df.iloc[i]['time'][3:])
            if real_wind[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 15:
                crash_wind = True
                plt.suptitle('Crash on wind:  Day: %d, city: %d, starting hour: %d, hour: %d, min: %d, xid: %d, yid : %d with cost: %.2f, mean cost: %2.f' %
                             (day, goal_city, starting_hour, hour_prev_step, min, df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, real_wind[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1],mean_weather[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1]))
                plt.scatter(df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, c='r', s=10)
            elif real_rainfall[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 4.0:
                crash_rainfall = True
                plt.suptitle('Crash on rainfall: Day: %d, city: %d, starting hour: %d, hour: %d, min: %d, xid: %d, yid : %d with cost: %.2f, mean cost: %2.f'
                    % (day, goal_city, starting_hour, hour_prev_step, min, df.iloc[i]['yid'] - 1,
                       df.iloc[i]['xid'] - 1, real_rainfall[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1],
                       mean_weather[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1]))
                plt.scatter(df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, c='r', s=10)

            if crash_wind or crash_rainfall:
                plt.waitforbuttonpress(0)
                break

    if not (crash_wind or crash_rainfall):
        return True
    else:
        return False


def reinforcement_learning_solution(cf):
    """
    This is a RL algorithm:
    The whole diagram is expanded with a third dimension T which has length 18*30 = 540
    :param cf:
    :return:
    """
    # we use A -star algorithm for deciding when to stop running the model
    # get the city locations
    cf.debug_draw = True
    cf.go_to_all_dir = True
    cf.day_list = [3]
    cf.goal_city_list = [9]
    cf.risky = False
    cf.model_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    print("Read Precompute A star route...")
    A_star_model_precompute_csv = load_a_star_precompute(cf)
    if cf.debug_draw:
        # draw figure maximum
        plt.figure(1)
        plt.clf()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

    start_time = timer()
    for day in cf.day_list:
        wind_real_day_hour_total = np.zeros(shape=(len(cf.model_number), cf.grid_world_shape[0], cf.grid_world_shape[1], int(cf.time_length)))
        for hour in range(3, 21):
            if day < 6:  # meaning this is a training day
                if cf.use_real_weather:
                    weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                    wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
                else:
                    wind_real_day_hour_temp = []
                    for model_number in cf.model_number:
                        # we average the result
                        weather_name = 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                        wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                        wind_real_day_hour_temp.append(wind_real_day_hour_model)
                    wind_real_day_hour = np.asarray(wind_real_day_hour_temp)
            else:
                wind_real_day_hour_temp = []
                for model_number in cf.model_number:
                    # we average the result
                    weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                    wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                    wind_real_day_hour_temp.append(wind_real_day_hour_model)
                wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
                wind_real_day_hour = np.mean(wind_real_day_hour_temp, axis=0)

            # we replicate the weather for the whole hour
            wind_real_day_hour[wind_real_day_hour >= cf.wall_wind] = cf.strong_wind_penalty_coeff
            if cf.risky:
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] = 1  # Every movement will have a unit cost
            elif cf.wind_exp:
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] -= cf.wind_exp_mean  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] /= cf.wind_exp_std  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] = np.exp(wind_real_day_hour[wind_real_day_hour < cf.wall_wind]).astype(
                    'int')  # with int op. if will greatly enhance the computatinal speed
            else:
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] /= cf.risky_coeff  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] += 1

            wind_real_day_hour_total[:, :, :, (hour - 3) * 30:(hour - 2) * 30] = wind_real_day_hour[:, :, :, np.newaxis]  # we replicate the hourly data

        for goal_city in cf.goal_city_list:
            city_start_time = timer()
            start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
            goal_loc = (int(city_data_df.iloc[goal_city]['xid']) - 1, int(city_data_df.iloc[goal_city]['yid']) - 1)
            # initiate a star 3d search
            # start location is the first time
            start_loc_3D = (start_loc[0], start_loc[1], 0)
            # the goal location spans from all the time stamps--> as long as we reach the goal in any time stamp,
            # we say we have reached the goal
            goal_loc_3D = [(goal_loc[0], goal_loc[1], t) for t in range(cf.time_length)]

            # We collect trajectories from all wind model
            go_to_all = extract_a_star(A_star_model_precompute_csv, day, goal_city)
            steps_a_star_all = [len(x) for x in go_to_all]
            # construct the 3d maze
            maze = Maze_3D(height=cf.grid_world_shape[0],
                           width=cf.grid_world_shape[1],
                           time_length=int(cf.time_length),
                           start_state=start_loc_3D,
                           goal_states=goal_loc_3D,
                           return_to_start=cf.return_to_start,
                           reward_goal=cf.reward_goal,
                           reward_move=cf.reward_move,
                           reward_obstacle=cf.reward_obstacle,
                           maxSteps=cf.maxSteps,
                           wind_real_day_hour_total=wind_real_day_hour_total,
                           cf=cf)

            model = Dyna_3D(rand=np.random.RandomState(cf.random_state),
                            maze=maze,
                            epsilon=cf.epsilon,
                            gamma=cf.gamma,
                            planningSteps=cf.planningSteps,
                            qLearning=cf.qLearning,
                            expected=cf.expected,
                            alpha=cf.alpha,
                            priority=cf.priority,
                            theta=cf.theta,
                            policy_init=go_to_all,
                            optimal_length_relax=cf.optimal_length_relax,
                            heuristic=cf.heuristic)

            start_time = timer()
            # track steps / backups for each episode
            steps = []

            print('Play for using A star trajectory with with model wind')
            num_episode = 0
            success_flag = False
            while num_episode < cf.a_star_loop or not success_flag:
                # if num_episode >= cf.a_star_loop:
                #     model.qlearning = False
                #     model.expected = True
                #     model.policy_init = []
                #     model.epsilon = 0.05
                #     model.maze.reward_obstacle = -1
                # if num_episode > 1000:
                #     break
                num_episode += 1
                model.maze.wind_model = random.choice(range(10))
                model.a_star_model = random.choice(range(10))
                print("A star model: %d, wind model: %d" % (model.maze.wind_model, model.a_star_model))
                steps.append(model.play(environ_step=True))
                # print best action w.r.t. current state-action values
                # printActions(currentStateActionValues, maze)
                # check whether the (relaxed) optimal path is found
                came_from, currentState, success_flag = model.checkPath(np.mean(steps_a_star_all))
                if success_flag:
                    print('Success in ep: %d, with %d steps' % (num_episode, len(came_from)))
                else:
                    print('Fail in ep: %d, with %d steps' % (num_episode, len(came_from)))

            print('Finish, using %.2f sec!, updating %d steps.' % (timer() - start_time, np.sum(steps)))
            route_list = []
            current_loc = list(set(goal_loc_3D) & set([currentState]))
            find_loc = current_loc[0]
            while came_from[find_loc] is not None:
                prev_loc = came_from[find_loc]
                route_list.append(prev_loc)
                find_loc = prev_loc

            # we reverse the route for plotting and saving
            route_list.reverse()
            if cf.debug_draw:
                for hour in range(3, 21):
                    if day < 6:  # meaning this is a training day
                        weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                        title = weather_name[:-4]
                        wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
                    else:
                        wind_real_day_hour_temp = []
                        for model_number in cf.model_number:
                            # we average the result
                            weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                            wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                            wind_real_day_hour_temp.append(wind_real_day_hour_model)
                        wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
                        wind_real_day_hour = np.mean(wind_real_day_hour_temp, axis=0)
                        weather_name = 'Test_forecast_wind_model_count_%d_day_%d_hour_%d.npy' % (len(cf.model_number), day, hour)

                    plt.clf()
                    plt.imshow(wind_real_day_hour, cmap=cf.colormap)
                    plt.colorbar()
                    ########## Plot all a-star
                    for m, go_to in enumerate(go_to_all):
                        for p in go_to.keys():
                            plt.scatter(p[1], p[0], c='black', s=1)
                        #anno_point = int(len(go_to.keys()) * (1+m) / 10)
                        anno_point = int(len(go_to.keys()) /2 )
                        go_to_sorted = sorted(go_to)
                        p = go_to_sorted[anno_point]
                        #plt.annotate(str(m+1), xy=(p[1], p[0]), color='indigo', fontsize=20)

                    # we also plot the city location
                    for idx in range(city_data_df.index.__len__()):
                        x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                        y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                        cid = int(city_data_df.iloc[idx]['cid'])
                        plt.scatter(y_loc, x_loc, c='r', s=40)
                        if wind_real_day_hour[x_loc, y_loc] >= cf.wall_wind:
                            plt.annotate(str(cid), xy=(y_loc, x_loc), color='black', fontsize=20)
                        else:
                            plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)
                    # we also draw some contours
                    x = np.arange(0, wind_real_day_hour.shape[1], 1)
                    y = np.arange(0, wind_real_day_hour.shape[0], 1)
                    X, Y = np.meshgrid(x, y)
                    CS = plt.contour(X, Y, wind_real_day_hour, (15,), colors='k')

                    for h in range(3, hour+1):
                        for p in route_list[(h - 3) * 30:(h - 2) * 30]:
                            plt.scatter(p[1], p[0], c=cf.colors[np.mod(h, 2)], s=10)
                            if h >= hour and wind_real_day_hour[p[0], p[1]] >= cf.wall_wind:
                                title = weather_name[:-4] + '.  Crash at: ' + str(p) + ' in hour: %d' % (p[2] / 30 + 3)
                                print(title)
                    for p in route_list:
                        plt.scatter(p[1], p[0],  c='yellow', s=1)

                    plt.clabel(CS, inline=1, fontsize=10)
                    plt.title(title)
                    plt.waitforbuttonpress(0.01)
                    if 30*(hour-2) > len(route_list):
                        break

            print('We reach the goal for day: %d, city: %d with: %d steps, using %.2f sec!' %
                  (day, goal_city, len(route_list), timer() - city_start_time))

    print('Finish writing submission, using %.2f sec!' % (timer() - start_time))


def reinforcement_learning_solution_new(cf):

    """
    This is a RL algorithm:
    The whole diagram is expanded with a third dimension T which has length 18*30 = 540
    :param cf:
    :return:
    """
    # we use A -star algorithm for deciding when to stop running the model
    # get the city locations
    cf.day_list = [3]
    cf.goal_city_list = [3]
    cf.risky = False
    cf.model_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    print("Read Precompute A star route...")
    A_star_model_precompute_csv = load_a_star_precompute(cf)

    for day in cf.day_list:
        print("Processing wind data...")
        wind_real_day_hour_total = process_wind(cf, day)

        for goal_city in cf.goal_city_list:
            city_start_time = timer()
            start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
            goal_loc = (int(city_data_df.iloc[goal_city]['xid']) - 1, int(city_data_df.iloc[goal_city]['yid']) - 1)
            # initiate a star 3d search
            # start location is the first time
            start_loc_3D = (start_loc[0], start_loc[1], 0)
            # the goal location spans from all the time stamps--> as long as we reach the goal in any time stamp,
            # we say we have reached the goal
            goal_loc_3D = [(goal_loc[0], goal_loc[1], t) for t in range(cf.time_length)]

            # We collect trajectories from all wind model
            go_to_all = extract_a_star(A_star_model_precompute_csv, day, goal_city)
            steps_a_star_all_mean = np.array([len(x) for x in go_to_all]).mean()
            # construct the 3d maze

            model = initialise_maze_and_model(cf, start_loc_3D, goal_loc_3D, wind_real_day_hour_total)
            start_time = timer()
            # track steps / backups for each episode
            steps = []
            print('Play for using A star trajectory with with model wind')

            total_Q = np.zeros(len(cf.model_number))
            total_reward = np.zeros(len(cf.model_number))
            action_value_function_sum = np.zeros(len(cf.model_number))
            model.epsilon = 0
            model.policy_init = go_to_all
            model.maze.wall_wind = np.inf  # we will ignore strong wind penalty for the moment
            model.qlearning = True
            model.expected = False
            # gamma is dependent upon total length, the more lengthy the route is, the small the gamma
            model.gamma = 1 + 0.01 * (steps_a_star_all_mean - 540) / 540.

            for m in range(len(cf.model_number)):
                model.maze.wind_model = m
                model.a_star_model = m
                print("A star model: %d, wind model: %d" % (model.maze.wind_model, model.a_star_model))
                steps.append(model.play(environ_step=True))
                # check whether the (relaxed) optimal path is found
                came_from, currentState, success_flag, total_Q[m], total_reward[m], action_value_function_sum[m]\
                    = model.checkPath(steps_a_star_all_mean, set_wind_to_zeros=False)
                print('Success in ep: %d, with %d steps. Q: %.2f' % (m, len(came_from), action_value_function_sum[m]))

            # Plot function
            # plot_state_action_value(model, city_data_df, cf)
            # how many times we will loop is depend upon the minimum length
            # a_star_loop = model.heuristic_fn(model.maze.START_STATE, model.maze.GOAL_STATES)
            a_star_loop = int(steps_a_star_all_mean)
            model.maze.c_baseline = cf.c_baseline_start
            c_baseline_step = (cf.c_baseline_start - cf.c_baseline_end) / a_star_loop

            success_flag = False
            save_length = int(a_star_loop * cf.optimal_length_relax) + 1
            total_Q_new = np.zeros(save_length)
            total_reward_new = np.zeros(save_length)
            action_value_function_sum_new = np.zeros(save_length)
            num_episode = 0
            model.policy_init = []
            alpha_step = (cf.alpha_start - cf.alpha_end) / a_star_loop
            epsilon_step = (cf.epsilon_start - cf.epsilon_end) / a_star_loop
            model.epsilon = cf.epsilon_start
            model.alpha = cf.alpha_start
            model.qlearning = cf.qLearning
            model.expected = cf.expected
            model.maze.risky = False
            model.maze.wall_wind = cf.wall_wind   # restore the penalty for the wind
            model.planningSteps = model.heuristic_fn(model.maze.START_STATE, model.maze.GOAL_STATES)
            model.double = cf.double
            # we need to copy the double Q-learning stateActionValue here
            model.stateActionValues2 = model.stateActionValues.copy()

            # polynomial learning rate for alpha
            if cf.polynomial_alpha:
                model.polynomial_alpha = cf.polynomial_alpha
                model.polynomial_alpha_coefficient = cf.polynomial_alpha_coefficient
                if not cf.double:
                    model.alpha_count = np.zeros(model.stateActionValues.shape).astype(int)
                else:
                    model.alpha_count_1 = np.zeros(model.stateActionValues.shape).astype(int)
                    model.alpha_count_2 = np.zeros(model.stateActionValues.shape).astype(int)

            while num_episode < a_star_loop or not success_flag:
                if num_episode >= a_star_loop * cf.optimal_length_relax:
                    break
                model.maze.wind_model = model.rand.choice(range(len(cf.model_number)))
                steps.append(model.play(environ_step=True))
                # check whether the (relaxed) optimal path is found
                came_from, currentState, success_flag, total_Q_new[num_episode], total_reward_new[num_episode], action_value_function_sum_new[num_episode]\
                    = model.checkPath(steps_a_star_all_mean, set_wind_to_zeros=False)
                if success_flag:
                    str2 = 'Success in ep: %d, with %d steps. sum(Q): %.2f, R: %.4f' % (num_episode, len(came_from), action_value_function_sum_new[num_episode], total_reward_new[num_episode])
                else:
                    str2 = 'Fail in ep: %d, with %d steps' % (num_episode, len(came_from))
                num_episode += 1
                model.epsilon = max(cf.epsilon_end, model.epsilon - epsilon_step)
                model.alpha = max(cf.alpha_end, model.alpha - alpha_step)   # we don't want our learning rate to be too large
                model.maze.c_baseline = max(cf.c_baseline_end, model.maze.c_baseline - c_baseline_step)
                # we also require that the model should traverse every state action more than
                if np.mean(np.array(model.alpha_action)) > 1./model.planningSteps**cf.polynomial_alpha_coefficient:
                    success_flag = False
                if cf.polynomial_alpha:
                    str1 = "Day: %d, City: %d, Episode %d/%d, wind model: %d, baseline: %2f, alpha(mean): %2f, epsilon: %2f. ### " % (day, goal_city, num_episode, a_star_loop, model.maze.wind_model+1, model.maze.c_baseline, np.mean(np.array(model.alpha_action)), model.epsilon)
                else:
                    str1 = "Day: %d, City: %d, Episode %d/%d, wind model: %d, baseline: %2f, alpha: %2f, epsilon: %2f. ### " % (day, goal_city, num_episode, a_star_loop, model.maze.wind_model+1, model.maze.c_baseline, model.alpha, model.epsilon)
                print(str1 + str2)

            print('Finish, using %.2f sec!, updating %d steps.' % (timer() - start_time, np.sum(steps)))
            route_list = []
            current_loc = list(set(goal_loc_3D) & set([currentState]))
            find_loc = current_loc[0]
            while came_from[find_loc] is not None:
                prev_loc = came_from[find_loc]
                route_list.append(prev_loc)
                find_loc = prev_loc

            # we reverse the route for plotting and saving
            route_list.reverse()
            if cf.debug_draw:
                draw_predicted_route(cf, day, go_to_all, city_data_df, route_list)
            else:
                flag_success = print_predicted_route(cf, day, go_to_all, city_data_df, route_list)

            if flag_success:
                print('We reach the goal for day: %d, city: %d with: %d steps, using %.2f sec!' %
                      (day, goal_city, len(route_list), timer() - city_start_time))

    print('Finish! using %.2f sec!' % (timer() - start_time))


def reinforcement_learning_solution_worker(cf, day, goal_city, A_star_model_precompute_csv):
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    wind_real_day_hour_total = process_wind(cf, day)

    city_start_time = timer()
    start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
    goal_loc = (int(city_data_df.iloc[goal_city]['xid']) - 1, int(city_data_df.iloc[goal_city]['yid']) - 1)
    # start location is the first time
    start_loc_3D = (start_loc[0], start_loc[1], 0)
    goal_loc_3D = [(goal_loc[0], goal_loc[1], t) for t in range(cf.time_length)]

    # We collect trajectories from all wind model
    go_to_all = extract_a_star(A_star_model_precompute_csv, day, goal_city)
    steps_a_star_all_mean = np.array([len(x) for x in go_to_all]).mean()
    # construct the 3d maze, we need Q Learning to initiate the value
    # cf.qlearning = True
    # cf.expected = False
    model = initialise_maze_and_model(cf, start_loc_3D, goal_loc_3D, wind_real_day_hour_total)
    steps = []
    model.epsilon = 0
    model.policy_init = go_to_all
    model.maze.wall_wind = np.inf  # we will ignore strong wind penalty for the moment
    model.qlearning = True
    model.expected = False
    model.double = False
    model.gamma = 1 + 0.01 * (steps_a_star_all_mean - 540) / 540.

    # A star model initialisation
    for m in range(len(cf.model_number)):
        model.maze.wind_model = m
        model.a_star_model = m
        steps.append(model.play(environ_step=True))

    a_star_loop = int(steps_a_star_all_mean)
    model.maze.c_baseline = cf.c_baseline_start
    c_baseline_step = (cf.c_baseline_start - cf.c_baseline_end) / a_star_loop

    success_flag = False
    save_length = int(a_star_loop * cf.optimal_length_relax) + 1
    total_Q_new = np.zeros(save_length)
    total_reward_new = np.zeros(save_length)
    action_value_function_sum_new = np.zeros(save_length)
    num_episode = 0
    model.policy_init = []
    model.alpha = cf.alpha_start
    epsilon_step = (cf.epsilon_start - cf.epsilon_end) / a_star_loop
    alpha_step = (cf.alpha_start - cf.alpha_end) / a_star_loop
    model.epsilon = cf.epsilon_start
    # using Expected sarsa to refine
    model.qlearning = cf.qLearning
    model.expected = cf.expected
    model.maze.risky = False
    model.maze.wall_wind = cf.wall_wind   # restore the penalty for the wind
    model.planningSteps = model.heuristic_fn(model.maze.START_STATE, model.maze.GOAL_STATES)
    # Double learning
    model.double = cf.double
    model.stateActionValues2 = model.stateActionValues.copy()
    # polynomial learning rate for alpha
    if cf.polynomial_alpha:
        model.polynomial_alpha = cf.polynomial_alpha
        model.polynomial_alpha_coefficient = cf.polynomial_alpha_coefficient
        if not cf.double:
            model.alpha_count = np.zeros(model.stateActionValues.shape).astype(int)
        else:
            model.alpha_count_1 = np.zeros(model.stateActionValues.shape).astype(int)
            model.alpha_count_2 = np.zeros(model.stateActionValues.shape).astype(int)

    # weather information fusion
    while num_episode <= a_star_loop or not success_flag:
        if num_episode >= a_star_loop * cf.optimal_length_relax:
            break
        # check whether the (relaxed) optimal path is found
        model.maze.wind_model = model.rand.choice(range(len(cf.model_number)))
        steps.append(model.play(environ_step=True))

        came_from, currentState, success_flag, total_Q_new[num_episode], total_reward_new[num_episode], \
        action_value_function_sum_new[num_episode] \
            = model.checkPath(steps_a_star_all_mean, set_wind_to_zeros=False)
        num_episode += 1
        model.epsilon = max(cf.epsilon_end, model.epsilon - epsilon_step)
        model.alpha = max(cf.alpha_end, model.alpha - alpha_step)  # we don't want our learning rate to be too large
        model.maze.c_baseline = max(cf.c_baseline_end, model.maze.c_baseline - c_baseline_step)
        # we also require that the model should traverse every state action more than
        if np.mean(np.array(model.alpha_action)) > 1. / model.planningSteps ** cf.polynomial_alpha_coefficient:
            success_flag = False
    # check whether the (relaxed) optimal path is found
    route_list = []
    current_loc = list(set(goal_loc_3D) & set([currentState]))
    if len(current_loc):
        find_loc = current_loc[0]
        while came_from[find_loc] is not None:
            prev_loc = came_from[find_loc]
            route_list.append(prev_loc)
            find_loc = prev_loc
        # we reverse the route for plotting and saving
        route_list.reverse()
    else:
        # if we didn't find a route, we choose the A* shortest path
        a_star_shortest_index = np.argmin(np.array([len(x) for x in go_to_all]))
        go_to_A_star_min = go_to_all[a_star_shortest_index]
        route_list.append(start_loc_3D)
        current_loc = start_loc_3D
        while current_loc in go_to_A_star_min.keys():
            next_loc = go_to_A_star_min[current_loc]
            current_loc = (next_loc[0], next_loc[1], next_loc[2]+1)
            route_list.append(current_loc)
        route_list.pop()  # for a_star_submission_3d, we don't attach goal

    sub_df = a_star_submission_3d(day, goal_city, goal_loc, route_list)
    csv_file_name = cf.csv_file_name[:-4] + '_day: %d, city: %d' % (day, goal_city) + '.csv'
    sub_df.to_csv(csv_file_name, header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])
    print('We reach the goal for day: %d, city: %d with: %d steps, using %.2f sec!' % (day, goal_city, len(route_list), timer() - city_start_time))
    sys.stdout.flush()
    return


def reinforcement_learning_solution_multiprocessing(cf):
    """
    This is a RL algorithm:
    The whole diagram is expanded with a third dimension T which has length 18*30 = 540
    :param cf:
    :return:
    """
    # we use A -star algorithm for deciding when to stop running the model
    # get the city locations
    start_time = timer()

    # when debugging concurrenty issues, it can be useful to have access to the internals of the objects provided by multiprocessing.
    multiprocessing.log_to_stderr()
    print("Read Precompute A star route...")
    print(help(cf))
    A_star_model_precompute_csv = load_a_star_precompute(cf)

    jobs = []
    for day in cf.day_list:
        for goal_city in cf.goal_city_list:
            p = multiprocessing.Process(target=reinforcement_learning_solution_worker, args=(cf, day, goal_city, A_star_model_precompute_csv))
            jobs.append(p)
            p.start()
            # because of the memory constraint, we need to wait for the previous to finish to finish in order
            # to initiate another function...
            if len(jobs) > cf.num_threads:
                jobs[-cf.num_threads].join()

    # waiting for the all the job to finish
    for j in jobs:
        j.join()
    # sub_csv is for submission
    collect_csv_for_submission(cf)

    print('Finish writing submission, using %.2f sec!' % (timer() - start_time))


def reinforcement_learning_solution_wind_and_rainfall(cf):

    """
    This is a RL algorithm:
    The whole diagram is expanded with a third dimension T which has length 18*30 = 540
    :param cf:
    :return:
    """
    # we use A -star algorithm for deciding when to stop running the model
    # get the city locations
    cf.day_list = [2]
    cf.goal_city_list = [4]
    cf.risky = False
    cf.model_number = list(range(1, 12))

    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    print("Read Precompute A star route...")
    A_star_model_precompute_csv = load_a_star_precompute_wind_and_rainfall(cf)
    for day in cf.day_list:
        for goal_city in cf.goal_city_list:
            for m in list(range(1, 12)):
                print('day %d, city: %d, model: %d' % (day, goal_city, m))
                print(len(A_star_model_precompute_csv[day][goal_city][m]['city_data_hour_df']))

    for day in cf.day_list:
        for goal_city in cf.goal_city_list:
            print("Processing weather data...")
            start_hour = A_star_model_precompute_csv[day][goal_city][1]['start_hour']
            start_min = A_star_model_precompute_csv[day][goal_city][1]['start_min']
            time_length = int((cf.hour_unique[1] - start_hour + 1) * 30 - start_min/2)
            # deal weather data
            wind_real_day_hour_total, rainfall_real_day_hour_total = process_wind_and_rainfall(cf, day, start_hour)

            city_start_time = timer()
            start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
            goal_loc = (int(city_data_df.iloc[goal_city]['xid']) - 1, int(city_data_df.iloc[goal_city]['yid']) - 1)
            # initiate a star 3d search
            # start location is the first time
            start_loc_3D = (start_loc[0], start_loc[1], 0)
            # the goal location spans from all the time stamps--> as long as we reach the goal in any time stamp,
            # we say we have reached the goal
            goal_loc_3D = [(goal_loc[0], goal_loc[1], t) for t in range(time_length)]

            # We collect trajectories from all wind model
            go_to_all = extract_a_star_wind_and_rainfall(A_star_model_precompute_csv, day, goal_city)
            steps_a_star_all_mean = np.array([len(x) for x in go_to_all]).mean()
            # construct the 3d maze

            model = initialise_maze_and_model_wind_and_rainfall(cf, start_loc_3D, goal_loc_3D, wind_real_day_hour_total,
                                                                rainfall_real_day_hour_total, time_length, start_min)
            start_time = timer()
            # track steps / backups for each episode
            steps = []
            print('Play for using A star trajectory with with model wind')

            total_Q = np.zeros(len(cf.model_number))
            total_reward = np.zeros(len(cf.model_number))
            action_value_function_sum = np.zeros(len(cf.model_number))
            model.epsilon = 0
            model.policy_init = go_to_all
            model.maze.wall_wind = np.inf  # we will ignore strong wind penalty for the moment
            model.qlearning = True
            model.expected = False
            model.double = False
            # gamma is dependent upon total length, the more lengthy the route is, the small the gamma

            for m in range(len(cf.model_number)):
                model.maze.wind_model = m
                model.a_star_model = m
                print("A star model: %d, wind model: %d" % (model.maze.wind_model, model.a_star_model))
                steps.append(model.play(environ_step=True))
                # check whether the (relaxed) optimal path is found
                came_from, currentState, success_flag, total_Q[m], total_reward[m], action_value_function_sum[m]\
                    = model.checkPath(steps_a_star_all_mean, set_wind_to_zeros=False)
                print('Success in ep: %d, with %d steps. Q: %.2f' % (m, len(came_from), action_value_function_sum[m]))

            # Plot function
            # plot_state_action_value(model, city_data_df, cf)
            # how many times we will loop is depend upon the minimum length
            # a_star_loop = model.heuristic_fn(model.maze.START_STATE, model.maze.GOAL_STATES)
            a_star_loop = 100
            success_flag = False
            save_length = int(a_star_loop * cf.optimal_length_relax) + 1
            total_Q_new = np.zeros(save_length)
            total_reward_new = np.zeros(save_length)
            action_value_function_sum_new = np.zeros(save_length)
            num_episode = 0
            model.policy_init = []
            alpha_step = (cf.alpha_start - cf.alpha_end) / a_star_loop
            epsilon_step = (cf.epsilon_start - cf.epsilon_end) / a_star_loop
            model.epsilon = cf.epsilon_start
            model.alpha = cf.alpha_start
            model.qlearning = cf.qLearning
            model.expected = cf.expected
            model.maze.risky = False
            model.maze.wall_wind = cf.wall_wind   # restore the penalty for the wind
            model.planningSteps = int(model.heuristic_fn(model.maze.START_STATE, model.maze.GOAL_STATES) // 10)

            while num_episode < a_star_loop or not success_flag:
                if num_episode >= a_star_loop * cf.optimal_length_relax:
                    break
                model.maze.wind_model = model.rand.choice(range(len(cf.model_number)))
                steps.append(model.play(environ_step=True))
                # check whether the (relaxed) optimal path is found
                came_from, currentState, success_flag, total_Q_new[num_episode], total_reward_new[num_episode], action_value_function_sum_new[num_episode]\
                    = model.checkPath(steps_a_star_all_mean, set_wind_to_zeros=False)
                if success_flag:
                    str2 = 'Success in ep: %d, with %d steps. sum(Q): %.2f, R: %.4f' % (num_episode, len(came_from)-1, action_value_function_sum_new[num_episode], total_reward_new[num_episode])
                else:
                    str2 = 'Fail in ep: %d, with %d steps' % (num_episode, len(came_from))
                num_episode += 1
                model.epsilon = max(cf.epsilon_end, model.epsilon - epsilon_step)
                model.alpha = max(cf.alpha_end, model.alpha - alpha_step)   # we don't want our learning rate to be too large
                # we also require that the model should traverse every state action more than
                str1 = "Day: %d, City: %d, Episode %d/%d, wind model: %d, alpha: %2f, epsilon: %2f. ### " % (day, goal_city, num_episode, a_star_loop, model.maze.wind_model+1, model.alpha, model.epsilon)
                print(str1 + str2)

            print('Finish, using %.2f sec!, updating %d steps.' % (timer() - start_time, np.sum(steps)))
            route_list = []
            current_loc = list(set(goal_loc_3D) & set([currentState]))
            find_loc = current_loc[0]
            while came_from[find_loc] is not None:
                prev_loc = came_from[find_loc]
                route_list.append(prev_loc)
                find_loc = prev_loc

            # we reverse the route for plotting and saving
            route_list.reverse()
            sub_df = a_star_submission_3d(day, goal_city, start_hour, start_min, goal_loc, route_list)

            flag_success = print_predicted_route_wind_and_rainfall(cf, day, goal_city, sub_df)

            if flag_success:
                print('We reach the goal for day: %d, city: %d with: %d steps, using %.2f sec!' %
                      (day, goal_city, len(route_list), timer() - city_start_time))

    print('Finish! using %.2f sec!' % (timer() - start_time))


def load_a_star_precompute_wind_and_rainfall(cf):
    # we will read both the mean weather and the individual 10 predictions
    A_star_model_precompute_csv = {}
    for day in cf.day_list:
        A_star_model_precompute_csv[day] = {}
        for goal_city in cf.goal_city_list:
            A_star_model_precompute_csv[day][goal_city] = {}
            file_patterns = cf.A_star_csv_patterns % (day, goal_city)
            files = fnmatch.filter(os.listdir(cf.A_star_precompute_path), file_patterns)
            for f in files:
                _splits = f.split('_')
                model_number = int(_splits[8][1:-1])
                _splits = f.split(' ')
                start_hour = int(_splits[5].split(',')[0])
                start_min = int(_splits[7].split('.')[0])
                city_data_hour_df = pd.read_csv(os.path.join(cf.A_star_precompute_path, f), index_col=None,names=['target', 'date', 'time', 'xid', 'yid'])

                A_star_model_precompute_csv[day][goal_city][model_number] = {}
                A_star_model_precompute_csv[day][goal_city][model_number]['start_hour'] = start_hour
                A_star_model_precompute_csv[day][goal_city][model_number]['start_min'] = start_min
                A_star_model_precompute_csv[day][goal_city][model_number]['city_data_hour_df'] = city_data_hour_df

    # Now we load the mean weather data
    for day in cf.day_list:
        for goal_city in cf.goal_city_list:
            start_hour = A_star_model_precompute_csv[day][goal_city][model_number]['start_hour']
            start_min = A_star_model_precompute_csv[day][goal_city][model_number]['start_min']
            file_patterns = '*costsSigmoid_4_14.5__model_number_*_day: %d, city: %d, start_hour: %d, start_min: %d.csv' % (day, goal_city, start_hour, start_min)
            files = fnmatch.filter(os.listdir(cf.A_star_precompute_mean_path), file_patterns)
            # there should be only one file
            for f in files:
                city_data_hour_df = pd.read_csv(os.path.join(cf.A_star_precompute_mean_path, f), index_col=None,names=['target', 'date', 'time', 'xid', 'yid'])
                A_star_model_precompute_csv[day][goal_city][11] = {}
                A_star_model_precompute_csv[day][goal_city][11]['start_hour'] = start_hour
                A_star_model_precompute_csv[day][goal_city][11]['start_min'] = start_min
                A_star_model_precompute_csv[day][goal_city][11]['city_data_hour_df'] = city_data_hour_df

    return A_star_model_precompute_csv


def extract_a_star_wind_and_rainfall(A_star_model_precompute_csv, day, goal_city):
    go_to_all = []
    for model_number in A_star_model_precompute_csv[day][goal_city].keys():
        go_to = {}
        predicted_df_now = A_star_model_precompute_csv[day][goal_city][model_number]['city_data_hour_df']
        for t in range(len(predicted_df_now)-1):
            go_to[(predicted_df_now.iloc[t]['xid']-1, predicted_df_now.iloc[t]['yid']-1, t)] = \
                (predicted_df_now.iloc[t+1]['xid']-1, predicted_df_now.iloc[t+1]['yid']-1, t)
        go_to_all.append(go_to)

    return go_to_all


def initialise_maze_and_model_wind_and_rainfall(cf, start_loc_3D, goal_loc_3D, wind_real_day_hour_total,
                                                rainfall_real_day_hour_total, time_length, start_min):
    cost_matrix = set_cost(cf, wind_real_day_hour_total, rainfall_real_day_hour_total, time_length)

    maze = Maze_3D(height=cf.grid_world_shape[0],
                   width=cf.grid_world_shape[1],
                   time_length=int(time_length),
                   start_state=start_loc_3D,
                   goal_states=goal_loc_3D,
                   reward_goal=1000,
                   cost_matrix=cost_matrix,
                   cf=cf,
                   start_min=start_min)

    model = Dyna_3D(rand=np.random.RandomState(cf.random_state),
                    maze=maze,
                    epsilon=cf.epsilon,
                    gamma=cf.gamma,
                    planningSteps=int(time_length),
                    qLearning=cf.qLearning,
                    expected=cf.expected,
                    alpha=cf.alpha,
                    priority=cf.priority,
                    theta=cf.theta,
                    optimal_length_relax=cf.optimal_length_relax,
                    polynomial_alpha=False)
    return model


def set_cost(cf, wind_real_day_hour_total, rainfall_real_day_hour_total, time_length):
    def sigmoid_cost(cf, real_weather):
        return (-1) * cf.c1 * (1 / (1 + np.exp(-cf.c2 * (real_weather - cf.c3))))
    real_weather = np.minimum(np.maximum(wind_real_day_hour_total, rainfall_real_day_hour_total * 15. / 4), 30)
    cost_matrix = np.zeros(shape=real_weather.shape)
    if cf.costs_linear:
        cost_matrix[real_weather >= cf.wall_wind] = -1.0 * time_length
        cost_matrix[real_weather < cf.wall_wind] = -1.0 * real_weather[real_weather < cf.wall_wind] / cf.wall_wind
    elif cf.costs_sigmoid:
        cost_matrix = sigmoid_cost(cf, real_weather)

    return cost_matrix


def reinforcement_learning_solution_multiprocessing_wind_and_rainfall(cf):
    """
    This is a RL algorithm:
    :param cf:
    :return:
    """
    # we use A -star algorithm for deciding when to stop running the model
    # get the city locations
    start_time = timer()
    print("Read Precompute A star route...")
    # when debugging concurrenty issues, it can be useful to have access to the
    # internals of the objects provided by multiprocessing.
    multiprocessing.log_to_stderr()
    print("Read Precompute A star route...")
    print(help(cf))
    A_star_model_precompute_csv = load_a_star_precompute_wind_and_rainfall(cf)
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))

    jobs = []
    for day in cf.day_list:
        for goal_city in cf.goal_city_list:
            p = multiprocessing.Process(target=reinforcement_learning_solution_worker_wind_and_rainfall,
                                        args=(cf, day, goal_city, A_star_model_precompute_csv, city_data_df))
            jobs.append(p)
            p.start()
            # because of the memory constraint, we need to wait for the previous to finish to finish in order
            # to initiate another function...
            if len(jobs) > cf.num_threads:
                jobs[-cf.num_threads].join()

    # waiting for the all the job to finish
    for j in jobs:
        j.join()
    # sub_csv is for submission
    collect_csv_for_submission(cf)

    print('Finish writing submission, using %.2f sec!' % (timer() - start_time))


def reinforcement_learning_solution_worker_wind_and_rainfall(cf, day, goal_city, A_star_model_precompute_csv, city_data_df):
    cf.model_number = list(range(1, 12))
    start_hour = A_star_model_precompute_csv[day][goal_city][1]['start_hour']
    start_min = A_star_model_precompute_csv[day][goal_city][1]['start_min']
    time_length = int((cf.hour_unique[1] - start_hour + 1) * 30 - start_min / 2)
    wind_real_day_hour_total, rainfall_real_day_hour_total = process_wind_and_rainfall(cf, day, start_hour)

    city_start_time = timer()
    start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
    goal_loc = (int(city_data_df.iloc[goal_city]['xid']) - 1, int(city_data_df.iloc[goal_city]['yid']) - 1)
    # start location is the first time
    start_loc_3D = (start_loc[0], start_loc[1], 0)
    goal_loc_3D = [(goal_loc[0], goal_loc[1], t) for t in range(cf.time_length)]

    # We collect trajectories from all wind model
    go_to_all = extract_a_star_wind_and_rainfall(A_star_model_precompute_csv, day, goal_city)
    steps_a_star_all_mean = np.array([len(x) for x in go_to_all]).mean()
    # construct the 3d maze, we need Q Learning to initiate the value
    model = initialise_maze_and_model_wind_and_rainfall(cf, start_loc_3D, goal_loc_3D, wind_real_day_hour_total,
                                                        rainfall_real_day_hour_total, time_length, start_min)
    steps = []
    model.epsilon = 0
    model.policy_init = go_to_all
    model.maze.wall_wind = np.inf  # we will ignore strong wind penalty for the moment
    model.qlearning = True
    model.expected = False
    model.double = False
    model.gamma = 1 + 0.01 * (steps_a_star_all_mean - time_length) / time_length

    # A star model initialisation
    for m in range(len(cf.model_number)):
        model.maze.wind_model = m
        model.a_star_model = m
        steps.append(model.play(environ_step=True))

    #a_star_loop = int(steps_a_star_all_mean)
    a_star_loop = int(steps_a_star_all_mean)
    success_flag = False
    save_length = int(a_star_loop * cf.optimal_length_relax) + 1
    total_Q_new = np.zeros(save_length)
    total_reward_new = np.zeros(save_length)
    action_value_function_sum_new = np.zeros(save_length)
    num_episode = 0
    model.policy_init = []
    model.alpha = cf.alpha_start
    epsilon_step = (cf.epsilon_start - cf.epsilon_end) / a_star_loop
    alpha_step = (cf.alpha_start - cf.alpha_end) / a_star_loop
    model.epsilon = cf.epsilon_start
    # using Expected sarsa to refine
    model.qlearning = cf.qLearning
    model.expected = cf.expected
    model.maze.risky = False
    model.maze.wall_wind = cf.wall_wind   # restore the penalty for the wind
    model.planningSteps = int(model.heuristic_fn(model.maze.START_STATE, model.maze.GOAL_STATES) // 10)
    # Double learning
    if cf.double:
        model.double = cf.double
        model.stateActionValues2 = model.stateActionValues.copy()

    # weather information fusion
    while num_episode <= a_star_loop or not success_flag:
        if num_episode >= a_star_loop * cf.optimal_length_relax:
            break
        # check whether the (relaxed) optimal path is found
        model.maze.wind_model = model.rand.choice(range(len(cf.model_number)))
        steps.append(model.play(environ_step=True))

        came_from, currentState, success_flag, total_Q_new[num_episode], total_reward_new[num_episode], action_value_function_sum_new[num_episode] \
            = model.checkPath(steps_a_star_all_mean, set_wind_to_zeros=False)
        num_episode += 1
        model.epsilon = max(cf.epsilon_end, model.epsilon - epsilon_step)
        model.alpha = max(cf.alpha_end, model.alpha - alpha_step)  # we don't want our learning rate to be too large

    # check whether the (relaxed) optimal path is found
    route_list = []
    current_loc = list(set(goal_loc_3D) & set([currentState]))
    if len(current_loc):
        find_loc = current_loc[0]
        while came_from[find_loc] is not None:
            prev_loc = came_from[find_loc]
            route_list.append(prev_loc)
            find_loc = prev_loc
        # we reverse the route for plotting and saving
        route_list.reverse()
    else:
        # if we didn't find a route, we choose the A* shortest path
        a_star_shortest_index = np.argmin(np.array([len(x) for x in go_to_all]))
        go_to_A_star_min = go_to_all[a_star_shortest_index]
        route_list.append(start_loc_3D)
        current_loc = start_loc_3D
        while current_loc in go_to_A_star_min.keys():
            next_loc = go_to_A_star_min[current_loc]
            current_loc = (next_loc[0], next_loc[1], next_loc[2]+1)
            route_list.append(current_loc)
        route_list.pop()  # for a_star_submission_3d, we don't attach goal

    sub_df = a_star_submission_3d(day, goal_city, start_hour, start_min, goal_loc, route_list)
    csv_file_name = cf.csv_file_name[:-4] + '_day: %d, city: %d' % (day, goal_city) + '.csv'
    sub_df.to_csv(csv_file_name, header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])
    print('We reach the goal for day: %d, city: %d with: %d steps, using %.2f sec!' % (day, goal_city, len(route_list), timer() - city_start_time))
    sys.stdout.flush()
    return

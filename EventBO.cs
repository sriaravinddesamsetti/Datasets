using System.Collections.Generic;
using System;

namespace MvcMovie.Models
{
    public class EventBO
    {
        public List<Event> GetEvents()
        {
            return new List<Event>
            {
                new Event("Sky Lantern Festival", "Stage Event", new DateTime(2020, 05, 01)),
                new Event("Rio Carnival", "Exhibition", new DateTime(2020, 04, 20)),
                new Event("Songkran", "Exhibition", new DateTime(2020, 04, 25)),
                new Event("Stars of the white house", "Stage Event", new DateTime(2020, 05, 15)),
                new Event("Cannes film festival", "Stage Event", new DateTime(2020, 04, 19)),
                new Event("Glastonbury", "Exhibition", new DateTime(2020, 03, 30)),
                new Event("Tomorrowland", "Exhibition", new DateTime(2020, 04, 24)),
                new Event("TrickEye", "Exhibition", new DateTime(2020, 04, 30)),
                new Event("Comic con International", "Stage Event", new DateTime(2020, 05, 05)),
            };
        }
    }
}
